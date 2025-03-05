import os
import torch
import itertools
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from .utils.data_utils import NormalizationType, save_dataset_statistics
from .rlds import dataset_statistics, build_interleaved_dataset

# NOTE: the Iterable dataset can'bt be selirized in distributed tranning due to pickle
class OpenXIterableDataset(IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        ds_name,
        data_root_dir,
        output_dir,
        data_mix,
        image_size=224,
        max_length=1024,
        is_train=True,
        data_augment=False,
        shuffle_buffer_size=1000_000,
        tsfm_thread_muti=1,
        read_thread_muti=1,
        obs_backward_steps=0,
        obs_backward_delta=1,
        action_forward_steps=0,
        use_raw_dataloader=False,
        fix_raw_length=None,
        statistic_exclude="no_exclude",
        normalized_statistic_path=None,
        vla_processor=None,
    ):
        super(OpenXIterableDataset, self).__init__()
        self.ds_name = ds_name
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.statistic_exclude = statistic_exclude
        self.use_raw_dataloader = use_raw_dataloader
        self.vla_processor = vla_processor
        self.image_size = image_size
        self.max_length = max_length
        self.is_train = is_train

        self.total_ranks = torch.distributed.get_world_size()
        self.current_rank = torch.distributed.get_rank()

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(os.path.join(self.data_mix, "1.0.0"), 1.0)]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        self.dataset_num = len(weights)
        self.rlds_config = dict(
            traj_transform_kwargs=dict(
                backward_windows_size=obs_backward_steps,  # If we wanted to feed / predict more than one step
                backward_delta=obs_backward_delta,
                forward_window_size=action_forward_steps,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=(self.image_size, self.image_size),
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec) * tsfm_thread_muti,
            traj_read_threads=len(mixture_spec) * read_thread_muti,
            train=self.is_train,
            shuffle_seed=3407 * self.current_rank,
            # shuffle_seed=42 * self.current_rank,
        )

        if data_augment:
            self.rlds_config["frame_transform_kwargs"].update(
                {
                    "image_augment_kwargs": dict(
                        random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                        random_brightness=[0.2],
                        random_contrast=[0.8, 1.2],
                        random_saturation=[0.8, 1.2],
                        random_hue=[0.05],
                        augment_order=[
                            "random_resized_crop",
                            "random_brightness",
                            "random_contrast",
                            "random_saturation",
                            "random_hue",
                        ],
                    )
                }
            )

        print("*** make rlds dataset...")
        self.rlds_dataset = None
        expected_length, self.dataset_statistics, self.sample_weights = dataset_statistics(**self.rlds_config)
        self.raw_length = expected_length * self.dataset_num

        # NOTE: in staget 2 ptraining, we use much less data, thus the resume'll stop immediately
        # set a fixed dataset length avoids the unexceptable traing interrupt
        if fix_raw_length:
            self.raw_length = fix_raw_length
            print(f"[Dataset] set a fixed dataset length {fix_raw_length} avoids the unexceptable traing interrupt!")

        if self.current_rank == 0:
            statistic = save_dataset_statistics(self.dataset_statistics, Path(output_dir) / "dataset_statistics.json")

        self.normalized_statistic_path = normalized_statistic_path

    def __len__(self):
        if self.use_raw_dataloader:
            return self.raw_length // self.total_ranks
        else:
            return self.raw_length

    def multi_modal_get_item(self, data_item):
        pixel_values_seq = []
        
        # TODO: add mutiple image inputs support (processor, model)
        for image_primary in data_item["observation"]["image_primary"]:  # (t h w c)
            image = Image.fromarray(image_primary)
            pixel_values_seq += [image] # [c h w]

        actions = torch.from_numpy(data_item["action"])  # (t e)
        lang = data_item["task"]["language_instruction"].lower()
        if isinstance(lang, bytes): lang = lang.decode()
        
        # TODO: move to processor
        ret = self.vla_processor(
            text=lang, 
            images=pixel_values_seq,
            suffix_actions=actions,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False, # do not normalize the image for zoe
        )

        model_inputs = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            token_type_ids=ret["token_type_ids"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=ret["pixel_values"],
            intrinsic=ret["intrinsic"],
            actions=actions,
        )
        return model_inputs

    def __iter__(self):
        if self.rlds_dataset is None:
            self.rlds_dataset = build_interleaved_dataset(
                weights=self.sample_weights, dataset_statistics=self.dataset_statistics, **self.rlds_config
            ).as_numpy_iterator()
            if torch.utils.data.get_worker_info() is not None:
                worker_total_num = torch.utils.data.get_worker_info().num_workers
                worker_id = torch.utils.data.get_worker_info().id
            else:
                worker_id = 0
                worker_total_num = 1
            self.rlds_dataset = itertools.islice(iter(self.rlds_dataset), worker_id, None, worker_total_num)

        for i, data_item in enumerate(self.rlds_dataset):
            ret = self.multi_modal_get_item(data_item)
            
            # print(f"rank: {self.current_rank}, \
            #         worker: {worker_id},  \
            #         iter_idx: {i},\
            #         dataset: {data_item['dataset_name']}, \
            #         traj_idx: {data_item['traj_index']}, \
            #         timestep: {data_item['observation']['timestep']}"
            # )

            if i < len(self):
                yield ret
            else:
                break


def build_datasets(
    data_args,
    output_dir,  # NOTE: from training_args.output_dir
    vla_processor=None,
) -> IterableDataset:
    train_dataset = OpenXIterableDataset(
        data_args.data_mix,
        data_args.data_root_dir,
        output_dir,
        data_args.data_mix,
        is_train=True,
        image_size=data_args.force_image_size,
        max_length=data_args.max_seq_length,
        shuffle_buffer_size=data_args.shuffle_buffer_size,
        tsfm_thread_muti=data_args.tsfm_thread_muti,
        read_thread_muti=data_args.read_thread_muti,
        obs_backward_steps=data_args.obs_backward_steps,
        obs_backward_delta=data_args.obs_backward_delta,
        action_forward_steps=data_args.action_forward_steps,
        use_raw_dataloader=data_args.use_raw_dataloader,
        fix_raw_length=data_args.fix_raw_length,
        statistic_exclude=data_args.statistic_exclude,
        normalized_statistic_path=data_args.normalized_statistic_path,
        data_augment=data_args.data_augment,
        vla_processor=vla_processor,
    )
    if data_args.train_only:
        eval_dataset = None
    else:
        eval_dataset = OpenXIterableDataset(
            data_args.data_mix,
            data_args.data_root_dir,
            output_dir,
            data_args.data_mix,
            is_train=False,
            image_size=data_args.force_image_size,
            max_length=data_args.max_seq_length,
            normalize_type=data_args.normalize_type,
            shuffle_buffer_size=data_args.shuffle_buffer_size,
            tsfm_thread_muti=data_args.tsfm_thread_muti,
            read_thread_muti=data_args.read_thread_muti,
            obs_backward_steps=data_args.obs_backward_steps,
            obs_backward_delta=data_args.obs_backward_delta,
            action_forward_steps=data_args.action_forward_steps,
            use_raw_dataloader=data_args.use_raw_dataloader,
            statistic_exclude=data_args.statistic_exclude,
            normalized_statistic_path=data_args.normalized_statistic_path,
            data_augment=data_args.data_augment,
            vla_processor=vla_processor,
        )

    return train_dataset, eval_dataset
