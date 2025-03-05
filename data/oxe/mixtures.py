"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "toto": [
        # ("bridge_oxe", 1.0),                                  # Version of Bridge V2 in Open-X GCP Bucket
        ("toto/0.1.0", 1.0),                                          # Original Version of Bridge V2 from Project Website
    ],

    "kuka": [
        ("kuka/0.1.0", 1.0),
    ],
    
    "droid": [
        ("droid/1.0.0", 1.0),
    ],

    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website

        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ],

    "rtx_franka": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
        ("taco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("viola", 1.0),
        ("toto", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("cmu_play_fusion", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data/0.1.0", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka/0.1.0", 0.8341046294),
        ("bridge_orig/1.0.0", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play/0.1.0", 2.0),
        ("jaco_play/0.1.0", 1.0),
        ("berkeley_cable_routing/0.1.0", 1.0),
        ("roboturk/0.1.0", 2.0),
        ("viola/0.1.0", 2.0),
        ("berkeley_autolab_ur5/0.1.0", 2.0),
        ("toto/0.1.0", 1.0),
        ("language_table/0.1.0", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds/0.1.0", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds/0.1.0", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds/0.1.0", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds/0.1.0", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds/0.1.0", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds/0.1.0", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0", 1.0),
        ("utaustin_mutex/0.1.0", 1.0),
        ("berkeley_fanuc_manipulation/0.1.0", 2.0),
        ("cmu_stretch/0.1.0", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z/0.1.0", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset/1.0.0", 1.0),
        ("dobbe/0.0.1", 0.2),
        ("droid/1.0.0", 0.06),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],
    
    "fractal": [
        ("fractal20220817_data/0.1.0", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],
    "bridge": [
        ("bridge_orig/1.0.0", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],
    
    # === Open-X Spatial Pretraining Dataset ===
    "oxe_spatial_vla_plus": [
        ("fractal20220817_data/0.1.0", 0.54087122203),                      # Google RT-1 Robot Data (Large-Scale)
        ("kuka/0.1.0", 0.4),                                                # NOTE: 0.8341046294 -> 0.4 due to lacking text prompts (no intrinsic)
        ("bridge_orig/1.0.0", 1.0),                                         # Original Version of Bridge V2 from Project Website
        ("taco_play/0.1.0", 2.0),
        ("jaco_play/0.1.0", 1.0),                                           # not so good
        ("berkeley_cable_routing/0.1.0", 1.0),
        ("roboturk/0.1.0", 2.0),
        ("viola/0.1.0", 2.0),
        ("berkeley_autolab_ur5/0.1.0", 2.0),
        ("toto/0.1.0", 0.5),                                                # NOTE: low quality, 1.0 -> 0.5
        ("language_table/0.1.0", 0.1),                                      # low quality
        ("stanford_hydra_dataset_converted_externally_to_rlds/0.1.0", 2.0), # no intrinsic
        ("austin_buds_dataset_converted_externally_to_rlds/0.1.0", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds/0.1.0", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0", 2.0),   # no intrinsic
        ("austin_sailor_dataset_converted_externally_to_rlds/0.1.0", 1.0),  # estimated intrinsic
        ("austin_sirius_dataset_converted_externally_to_rlds/0.1.0", 1.0),  # estimated intrinsic
        ("dlr_edan_shared_control_converted_externally_to_rlds/0.1.0", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0", 1.0), # low quality
        ("utaustin_mutex/0.1.0", 1.0),
        ("berkeley_fanuc_manipulation/0.1.0", 0.5),                         # NOTE: low quality, 2.0 -> 0.5
        ("cmu_stretch/0.1.0", 1.0),                                         # low quality
        ("bc_z/0.1.0", 0.2),                                                # NOTE: use v0.1.0 --> later versions broken
        ("fmb_dataset/1.0.0", 0.2),                                         # NOTE: 1.0 --> 0.2, right preference
        ("dobbe/0.0.1", 0.2),                                               # not so good
        ("droid/1.0.0", 0.06),                                              # NOTE: remove droid in stage2!
        ("rh20t_rlds/1.0.0", 0.015),
    ],

    "oxe_spatial_vla_plus_stage2": [
        ("fractal20220817_data/0.1.0", 0.54087122203),                      # Google RT-1 Robot Data (Large-Scale)
        ("kuka/0.1.0", 0.4),                                                # NOTE: 0.8341046294 -> 0.4 due to lacking text prompts (no intrinsic)
        ("bridge_orig/1.0.0", 1.0),                                         # Original Version of Bridge V2 from Project Website
        ("taco_play/0.1.0", 2.0),
        ("jaco_play/0.1.0", 1.0),                                           # not so good
        ("berkeley_cable_routing/0.1.0", 1.0),
        ("roboturk/0.1.0", 2.0),
        ("viola/0.1.0", 2.0),
        ("berkeley_autolab_ur5/0.1.0", 2.0),
        ("toto/0.1.0", 0.5),                                                # NOTE: low quality, 1.0 -> 0.5
        ("language_table/0.1.0", 0.1),                                      # low quality
        ("stanford_hydra_dataset_converted_externally_to_rlds/0.1.0", 2.0), # no intrinsic
        ("austin_buds_dataset_converted_externally_to_rlds/0.1.0", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds/0.1.0", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0", 2.0),   # no intrinsic
        ("austin_sailor_dataset_converted_externally_to_rlds/0.1.0", 1.0),  # estimated intrinsic
        ("austin_sirius_dataset_converted_externally_to_rlds/0.1.0", 1.0),  # estimated intrinsic
        ("dlr_edan_shared_control_converted_externally_to_rlds/0.1.0", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0", 1.0), # low quality
        ("utaustin_mutex/0.1.0", 1.0),
        ("berkeley_fanuc_manipulation/0.1.0", 0.5),                         # NOTE: low quality, 2.0 -> 0.5
        ("cmu_stretch/0.1.0", 1.0),                                         # low quality
        ("bc_z/0.1.0", 0.2),                                                # NOTE: use v0.1.0 --> later versions broken
        ("fmb_dataset/1.0.0", 0.2),                                         # NOTE: 1.0 --> 0.2, right preference
        ("dobbe/0.0.1", 0.2),                                               # not so good
        # ("droid/1.0.0", 0.06),                                              # NOTE: remove droid in stage2!
        ("rh20t_rlds/1.0.0", 0.015),
    ],
}
# fmt: on
OXE_EXCLUDE_MIXTURE: Dict[str, List[str]] = {
    "tokenizer_gaussian": [
        "language_table/0.1.0",
        "berkeley_fanuc_manipulation/0.1.0",
        "cmu_stretch/0.1.0",
        "jaco_play/0.1.0"
    ],
    "no_exclude": []
}