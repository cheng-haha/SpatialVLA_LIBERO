import json
import os

import torch
import torch.nn as nn
import datasets
from dataclasses import field, fields, make_dataclass
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer, logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.trainer import is_datasets_available, seed_worker, _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.training_args import TrainingArguments


logger = logging.get_logger(__name__)


def get_num_layer_for_vit_and_qllama(var_name, vit_num_max_layer, llama_num_max_layer):
    if var_name.startswith("internvl."):
        var_name = var_name[len("internvl.") :]
    if var_name in (
        "query_tokens",
        "logit_scale",
    ):
        return 0
    if var_name.startswith("clip_projector."):
        return vit_num_max_layer
    if var_name.startswith("clip_projector2.") or var_name.startswith("itm_head.") or var_name == "text_projection":
        return llama_num_max_layer
    if var_name.startswith("vision_model."):
        if "embeddings." in var_name:
            return 0
        if "layers." in var_name:
            var_name = var_name.split("layers.")[-1]
            layer_id = int(var_name.split(".")[0])
            return layer_id + 1
    if var_name.startswith("qllama."):
        if "embed_tokens" in var_name:
            return 0
        if "layers." in var_name:
            var_name = var_name.split("layers.")[-1]
            layer_id = int(var_name.split(".")[0])
            return layer_id + 1
        else:
            return llama_num_max_layer
    return 0


def param_classification(name):
    if name.startswith("internvl."):
        name = name[len("internvl.") :]
    if name in ["query_tokens", "text_projection", "logit_scale"]:
        return "qllama"
    elif name.startswith("vision_model."):
        return "vit"
    elif name.startswith("qllama."):
        return "qllama"
    elif name.startswith("clip_projector."):
        return "vit"
    elif name.startswith("clip_projector2."):
        return "qllama"
    elif name.startswith("itm_head."):
        return "qllama"
    else:
        return "other"


def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    parameter_groups = {}
    try:  # for stage2 model
        vit_num_layers = opt_model.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.config.qllama_config.num_hidden_layers + 2
    except:  # for stage3 model
        vit_num_layers = opt_model.internvl.config.vision_config.num_hidden_layers + 2
        qllama_num_layers = opt_model.internvl.config.qllama_config.num_hidden_layers + 2
    print("vit_num_layers:", vit_num_layers)
    print("qllama_num_layers:", qllama_num_layers)

    vit_layer_decay_rate = float(os.getenv("VIT_LAYER_DECAY_RATE", 1.0))
    qllama_layer_decay_rate = float(os.getenv("QLLAMA_LAYER_DECAY_RATE", 1.0))
    qllama_lr_scale = float(os.getenv("QLLAMA_LR_SCALE", 1.0))
    print("vit_layer_decay_rate:", vit_layer_decay_rate)
    print("qllama_layer_decay_rate:", qllama_layer_decay_rate)
    print("qllama_lr_scale:", qllama_lr_scale)

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = self.args.weight_decay

        cls = param_classification(name)
        layer_id = get_num_layer_for_vit_and_qllama(name, vit_num_layers, qllama_num_layers)
        group_name = "%s_layer_%d_%s" % (cls, layer_id, group_name)
        if group_name not in parameter_groups:
            if cls == "vit":
                scale = vit_layer_decay_rate ** (vit_num_layers - layer_id - 1)
            elif cls == "qllama":
                scale = qllama_layer_decay_rate ** (qllama_num_layers - layer_id - 1)
                scale = scale * qllama_lr_scale
            else:
                scale = 1.0
            scale = min(1.0, scale)
            parameter_groups[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale,
                "group_name": group_name,
                "lr": scale * self.args.learning_rate,
            }
        parameter_groups[group_name]["params"].append(param)
        parameter_groups[group_name]["param_names"].append(name)

        rank = torch.distributed.get_rank()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))

    optimizer_grouped_parameters = list(parameter_groups.values())
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped / 2 ** 20}M params")

    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp

        self.optimizer = smp.DistributedOptimizer(self.optimizer)

    return self.optimizer


def replace_create_optimizer():
    print("Replace original create_optimizer with custom create_optimizer")
    transformers.Trainer.create_optimizer = create_optimizer


def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    if train_dataset.use_raw_dataloader:
        return DataLoader(train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    print("Replace train dataloader!!")


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
        loss *= self.accelerator.num_processes

    with torch.no_grad():
        logits = outputs["logits"]  # (bs, seq, voc)
        labels = inputs["labels"]  # (bs, seq)
        shift_logits = logits[..., :-1, :].argmax(-1).contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = (shift_labels >= model.action_tokenizer.translation_tokenizer.token_start_idx) & (
            shift_labels <= model.action_tokenizer.gripper_tokenizer.token_end_idx
        )
        gt_action_ids, pred_action_ids = shift_labels[mask], shift_logits[mask]
        correct_preds = gt_action_ids == pred_action_ids
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # NOTE: acc of translation, rotation and gripper
        token_start_idx, token_end_idx = (
            model.action_tokenizer.translation_tokenizer.token_start_idx,
            model.action_tokenizer.translation_tokenizer.token_end_idx,
        )
        translation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            model.action_tokenizer.rotation_tokenizer.token_start_idx,
            model.action_tokenizer.rotation_tokenizer.token_end_idx,
        )
        rotation_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        token_start_idx, token_end_idx = (
            model.action_tokenizer.gripper_tokenizer.token_start_idx,
            model.action_tokenizer.gripper_tokenizer.token_end_idx,
        )
        gripper_mask = (gt_action_ids >= token_start_idx) & (gt_action_ids <= token_end_idx)

        translation_gt_action_ids, translation_pred_action_ids = gt_action_ids[translation_mask], pred_action_ids[translation_mask]
        rotation_gt_action_ids, rotation_pred_action_ids = gt_action_ids[rotation_mask], pred_action_ids[rotation_mask]
        gripper_gt_action_ids, gripper_pred_action_ids = gt_action_ids[gripper_mask], pred_action_ids[gripper_mask]

        translation_correct_preds = translation_gt_action_ids == translation_pred_action_ids
        rotation_correct_preds = rotation_gt_action_ids == rotation_pred_action_ids
        gripper_correct_preds = gripper_gt_action_ids == gripper_pred_action_ids

        translation_action_accuracy = translation_correct_preds.sum().float() / translation_mask.sum().float()
        rotation_action_accuracy = rotation_correct_preds.sum().float() / rotation_mask.sum().float()
        gripper_action_accuracy = gripper_correct_preds.sum().float() / gripper_mask.sum().float()

        # convert to continue actions
        gt_actions = inputs["actions"].reshape(-1, 7).to(device="cpu", dtype=torch.float32)
        pred_actions = model.action_tokenizer.decode_token_ids_to_actions(pred_action_ids.cpu().numpy().reshape(-1, 3))
        l1_loss = nn.functional.l1_loss(torch.tensor(pred_actions), torch.tensor(gt_actions))

        self.log(
            {
                "accuracy": action_accuracy.item(),
                "translation_accuracy": translation_action_accuracy.item(),
                "rotation_accuracy": rotation_action_accuracy.item(),
                "gripper_accuracy": gripper_action_accuracy.item(),
                "l1_loss": l1_loss.item(),
            }
        )

    return (loss, outputs) if return_outputs else loss


def replace_compute_loss():
    transformers.Trainer.compute_loss = compute_loss
    print("Replace compute_loss!!")


def add_fields_to_training_arguments(**kwargs):
    training_arguments = make_dataclass(
        "TrainingArguments",
        [(key, value[0], field(default=value[1])) for key, value in kwargs.items()] + [(f.name, f.type, f) for f in fields(TrainingArguments)],
        bases=(TrainingArguments,),
    )
    # SOLVE pickle: attribute lookup xxx on types failed
    training_arguments.__module__ = __name__
    # SOLVE pickle: it's not the same object as xxx
    globals()["TrainingArguments"] = training_arguments
    print("Add fields to TrainingArguments!!")
    return training_arguments