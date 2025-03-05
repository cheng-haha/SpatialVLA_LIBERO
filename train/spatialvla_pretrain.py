import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import torch
from torch import nn
import torch.distributed as dist
from train.dist_utils import init_dist

from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
    SphericalCoordinateActionTokenizer,
)
from patch import concat_pad_data_collator, replace_train_sampler

from train.trainer_monkey_patch import (
    replace_create_optimizer,
    replace_train_dataloader,
    replace_compute_loss,
)
from PIL import Image, ImageFile, PngImagePlugin
import transformers
from transformers import (
    AutoTokenizer,
    ZoeDepthConfig,
    ZoeDepthForDepthEstimation,
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from data.dataset import build_datasets

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# custom monkey patch
replace_train_dataloader()
replace_compute_loss()
replace_train_sampler()

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier for resume training."
        },
    )
    vision_zoe_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier for zoe model."
        },
    )
    vlm_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier for vlm model."
        },
    )
    use_vision_zoe: bool = field(
        default=True,
        metadata={"help": "Set to True to use vision zoe model."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the language model."},
    )
    freeze_llm_embed: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the LLM embeddings."},
    )
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    freeze_zoe: bool = field(
        default=True,
        metadata={"help": "Set to True to freeze the zoe model."},
    )
    freeze_projector: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the MLP layers of the model."},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={
            "help": "Specify the number of ViT layers to unfreeze. Default is 0."
        },
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={
            "help": "Specify the layer of ViT feature map to use. Default is last layer."
        },
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={
            "help": "Set the LoRA adapter rank for the backbone model. Default is 0."
        },
    )
    use_llm_lora: int = field(
        default=0,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={"help": "Set to True to enable the use of a custom trainer."},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    vision_attn_dropout: float = field(
        default=0.0,
        metadata={"help": "Set the drop path rate for the ViT model. Default is 0."},
    )
    ps_version: str = field(
        default="v1",
        metadata={
            "help": "Specify the version of pixel shuffle implementation. Default is `v1`."
            "Please use `v2` to fix the bug of transposed image."
        },
    )
    action_config: Path = field(
        default="scripts/action_config.json",
        metadata={"help": "path to the action config file."},
    )
    un_tie_weight: bool = field(
        default=False,
        metadata={"help": "Set to True to untie the weight of LLM and vision model."},
    )
    ego3d_as_pe: Optional[bool] = field(
        default=True, metadata={"help": "Use ego3d as position encoding."}
    )
    n_freqs: Optional[int] = field(
        default=8, metadata={"help": "Number of frequencies for ego3d."}
    )
    ego3d_patch_reso: Optional[int] = field(
        default=2, metadata={"help": "resoluation of ego3d."}
    )
    use_flash_attn2: bool = field(
        default=False,
        metadata={"help": "Set to True to untie the weight of LLM and vision model."},
    )
    min_sigma: float = field(
        default=0.0,
        metadata={"help": "Set the minimum sigma for creating action grids."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "The root directory of the dataset. Default is `data`."},
    )
    train_only: Optional[bool] = field(
        default="True",
        metadata={"help": "Whether to load eval dataset"},
    )
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "The name of the dataset mixture. Default is `bridge`."},
    )
    statistic_exclude: Optional[str] = field(
        default="no_exclude",
        metadata={
            "help": "The name of the dataset mixture excluded during tokenizer statistic. Default is `no_exclude`."
        },
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={"help": "Set the desired size for the image. Default is 224."},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Set the desired down-sampling ratio for the image. Default is 1.0."
        },
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use data resampling."},
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The noise_alpha value for NEFTune. Default is None."},
    )
    data_augment: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use image augmentation."},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=1000_000,
        metadata={
            "help": "The shuffle buffer size for the dataset. Default is 1000000."
        },
    )
    tsfm_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds transfom. Default is 1."},
    )
    read_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds reader. Default is 1."},
    )
    obs_backward_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of backward steps in observation. 0 indicates current"
        },
    )
    obs_backward_delta: Optional[int] = field(
        default=1, metadata={"help": "Backward delta in observation."}
    )
    action_forward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of forward steps in action. 0 indicates current"},
    )
    fix_raw_length: Optional[int] = field(
        default=None, metadata={"help": "fix the iterable dataset iter length."}
    )
    use_raw_dataloader: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use raw dataloader"}
    )
    intrinsic_config_path: Path = field(
        default="scripts/intrinsics.json",
        metadata={"help": "path to the intrinsic config file."},
    )
    normalized_statistic_path: Path = field(
        default=None,
        metadata={"help": "path to the normalized statistic file."},
    )


def main():
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        ckpt_files = list(
            filter(
                lambda x: x.startswith("checkpoint"),
                os.listdir(training_args.output_dir),
            )
        )
        if last_checkpoint is None and len(ckpt_files) > 0:
            ckpt_files = list(
                filter(
                    lambda x: x.startswith("checkpoint"),
                    os.listdir(training_args.output_dir),
                )
            )
        if last_checkpoint is None and len(ckpt_files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"🔥Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # NOTE: 1. initializing models and load tokenizer
    if model_args.model_name_or_path is not None:
        processor = SpatialVLAProcessor.from_pretrained(
            model_args.model_name_or_path, local_files_only=True
        )
        spatial_token_num = processor.action_tokenizer.vocab_size
        action_tokenizer = processor.action_tokenizer
        tokenizer = processor.tokenizer
        tokenizer.model_max_length = data_args.max_seq_length
    else:
        # Load pretrained model, tokenizer, and image processor
        action_config = json.load(open(model_args.action_config))
        spatial_token_num = action_config["num_bins"]["total"]
        print(f"will add {spatial_token_num} spatial action tokens")

        tokenizer_path = model_args.model_name_or_path or model_args.vlm_path
        logger.info(f"Loading Tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        tokenizer.tokenizer_path = tokenizer_path
        tokenizer.model_max_length = data_args.max_seq_length

    # load models
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    print(f"🔥 torch_dtype {torch_dtype}, {training_args.bf16}")
    if model_args.model_name_or_path is not None:
        logger.info("Loading SpatialVLA Model...")
        config = SpatialVLAConfig.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        config.use_spatial_token = model_args.freeze_llm_embed
        model = SpatialVLAForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if model_args.use_flash_attn2:
            model.language_model.config._attn_implementation = model.config.text_config._attn_implementation_internal = "flash_attention_2"
            model.vision_tower.config._attn_implementation = model.config.vision_config._attn_implementation_internal = "flash_attention_2"
    else:
        logger.info("Loading Vision Model...")
        """
        flash_attention_2: 
        https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/gemma2/modeling_gemma2.py#L402
        https://github.com/huggingface/transformers/blob/c8c8dffbe45ebef0a8dba4a51024e5e5e498596b/src/transformers/configuration_utils.py#L303C58-L303C77
        """
        # load paligemma model
        flash_attn_args = (
            {"attn_implementation": "flash_attention_2"}
            if model_args.use_flash_attn2
            else {}
        )
        paligemma_config = PaliGemmaConfig.from_pretrained(
            model_args.vlm_path,
            torch_dtype=torch_dtype,
            local_files_only=True,
            **flash_attn_args,
        )
        paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_args.vlm_path,
            config=paligemma_config,
            torch_dtype=torch_dtype,
            revision="bfloat16",
            local_files_only=True,
        )

        vision_zoe_config = ZoeDepthConfig.from_pretrained(
            model_args.vision_zoe_path, torch_dtype=torch_dtype, local_files_only=True
        )
        vision_zoe_model = ZoeDepthForDepthEstimation.from_pretrained(  # zoe does not support Flash Attention 2.0 yet.
            model_args.vision_zoe_path,
            config=vision_zoe_config,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

        logger.info("Building SpatialVLAConfig...")
        config = SpatialVLAConfig(
            **paligemma_config.to_dict(),
            vision_zoe_config=vision_zoe_config.to_dict(),
            spatial_token_num=spatial_token_num,
            use_spatial_token=model_args.freeze_llm_embed,
            ego3d_patch_reso=model_args.ego3d_patch_reso,
            n_freqs=model_args.n_freqs,
            use_vision_zoe=model_args.use_vision_zoe,
            **flash_attn_args,
        )

        logger.info("Building SpatialVLA Model...")
        model = SpatialVLAForConditionalGeneration(
            config,
            vision_model=paligemma_model.vision_tower,
            vision_zoe_model=vision_zoe_model,
            projector_model=paligemma_model.multi_modal_projector,
            language_model=paligemma_model.language_model,
        )

    # NOTE: 2. build datasets
    train_dataset, eval_dataset = build_datasets(
        data_args,
        training_args.output_dir,
        vla_processor=None,
    )

    # NOTE: 3. build action tokenizer, with action tokenizer in front of language tokenizer
    if model_args.model_name_or_path is None:
        gs_params = json.load(open(train_dataset.normalized_statistic_path))
        action_tokenizer = SphericalCoordinateActionTokenizer(
            tokenizer,
            num_bins=action_config["num_bins"],
            gs_params=gs_params,
            use_spherical=action_config["use_spherical"],
            min_sigma=model_args.min_sigma,
        )

        # NOTE: add action bin tokens: https://github.com/huggingface/transformers/issues/12632
        num_new_tokens = action_tokenizer.vocab_size
        assert num_new_tokens == spatial_token_num, "newly added tokens {num_new_tokens} must equal to pre caculated spatial_token_num {spatial_token_num}"

        # NOTE: modify language embedding and lm_head if train from scratch
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            output_embeddings = model.language_model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # NOTE: replace the llm head, freeze embedding tokens
        """https://github.com/meta-llama/llama-models/issues/172"""
        if model_args.un_tie_weight:
            model.config.text_config.tie_word_embeddings = False
            model.language_model.config.tie_word_embeddings = False
            new_lm_head = nn.Linear(
                model.config.text_config.hidden_size,
                model.config.text_config.vocab_size,
                bias=False,
            )
            new_lm_head.weight.data = (
                model.language_model.lm_head.weight.detach().clone()
            )
            model.language_model.lm_head = new_lm_head
    else:
        num_new_tokens = action_tokenizer.vocab_size

    # NOTE: overwrite attributes
    model.action_token_begin_idx = model.config.action_token_begin_idx = action_tokenizer.action_token_begin_idx
    # use_cache vs : https://discuss.huggingface.co/t/why-is-use-cache-incompatible-with-gradient-checkpointing/18811/6
    # model.config.text_config.use_cache = model.language_model.config.use_cache = False
    model.vision_tower.gradient_checkpointing = True

    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_llm_embed:
        print(
            f"embed_tokens.weight: {model.language_model.model.embed_tokens.weight}, spatial_embed_tokens.weight: {model.spatial_embed_tokens.weight}"
        )
        model.language_model.model.embed_tokens.weight.requires_grad = False
        model.spatial_embed_tokens.weight.data = (
            model.language_model.model.embed_tokens.weight.data[-num_new_tokens:]
        )

    if model_args.freeze_vision_tower:
        # model.vision_tower = model.vision_tower.eval()
        _freeze_params(model.vision_tower)

    if model_args.freeze_zoe and model_args.use_vision_zoe:
        model.vision_zoe_model = model.vision_zoe_model.eval()
        _freeze_params(model.vision_zoe_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        lm_head = (
            model.language_model.output
            if model.config.text_config.model_type == "internlm2"
            else model.language_model.lm_head
        )
        lm_head.weight.requires_grad = True
        if lm_head.bias is not None:
            lm_head.bias.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(
            r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora
        )
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(
            r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora
        )
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_projector:
        _freeze_params(model.multi_modal_projector)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_tower.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True

    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    set_seed(training_args.seed)

    if model_args.use_custom_trainer:
        replace_create_optimizer()

    print(f"**** max_grad_norm: {training_args.max_grad_norm}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator, # custom data collator
    )

    # NOTE: register for auto save and map
    SpatialVLAConfig.register_for_auto_class()
    SpatialVLAForConditionalGeneration.register_for_auto_class()
    SpatialVLAProcessor.register_for_auto_class()

    # NOTE：build processor
    statistic = json.load(open(f"{training_args.output_dir}/dataset_statistics.json"))
    if model_args.model_name_or_path is None:
        intrinsic_config = json.load(open(data_args.intrinsic_config_path))
        paligemma_processor = PaliGemmaProcessor.from_pretrained(
            model_args.vlm_path, local_files_only=True
        )
        paligemma_processor.image_processor.do_normalize = False  # NOTE: we nomalize in model, instead of processor
        processor = SpatialVLAProcessor(
            image_processor=paligemma_processor.image_processor,
            tokenizer=tokenizer,
            statistics=statistic,
            bin_policy=action_tokenizer.bin_policy,
            intrinsic_config=intrinsic_config,
            action_config=action_config,
            num_obs_steps=data_args.obs_backward_steps + 1,
            obs_delta=data_args.obs_backward_delta,
            action_chunk_size=data_args.action_forward_steps + 1,
        )
    else:
        processor.statistics.update(statistic)  # merge

    model.action_tokenizer = action_tokenizer
    train_dataset.vla_processor = processor
    if not data_args.train_only:
        eval_dataset.action_tokenizer = action_tokenizer
        eval_dataset.vla_processor = processor

    # NOTE：save processor for rapid from pretrained
    if dist.get_rank() == 0:
        processor.save_pretrained(training_args.output_dir)
        logger.info(f"save the preprocessor to {training_args.output_dir}")

        import shutil
        copy_files = ["model/action_tokenizer.py", "test/test_huggingface.py",]
        for file in copy_files:
            try:
                shutil.copy(file, training_args.output_dir)
            except:
                pass
        logger.info(f"copy files {copy_files} to {training_args.output_dir}")

        delete_files = [
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "tokenizer.json",
        ]
        for file in delete_files:
            try:
                os.remove(f"{training_args.output_dir}/{file}")
            except:
                pass
        logger.info(f"remove duplicate files {delete_files}")

    # NOTE: Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


class ProfilerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./profiler_output"
            ),
        )
        self.profiler.__enter__()

    def training_step(self, model, inputs):
        output = super().training_step(model, inputs)
        self.profiler.step()
        return output

    def __del__(self):
        self.profiler.__exit__(None, None, None)


if __name__ == "__main__":
    main()
