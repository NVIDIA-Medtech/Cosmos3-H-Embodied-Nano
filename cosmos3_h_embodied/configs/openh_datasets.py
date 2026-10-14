# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Open-H dataset mixture of the H16 task-prompt recipe.

Each family is one ``dataset_entry`` of :func:`~cosmos3_h_embodied.data.unified_dataset.wrap_dataset`.
Roots are paths relative to the prepared data directory (``OPENH_DATA_ROOT``), which mirrors
the layout of the Open-H Hugging Face dataset (see ``tools/data_prep/README.md``). Roots
marked ``public=False`` were part of the training mixture of the released checkpoint but
are not on the Hugging Face dataset; they are left out unless ``include_nonpublic=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cosmos3_h_embodied.data.history_dataset import CmrClinicalHistoryDataset, GeneralizedDvrkHistoryDataset
from cosmos3_h_embodied.data.unified_dataset import dataset_entry
from cosmos_framework.utils.lazy_config import LazyCall as L

DATA_ROOT = "${oc.env:OPENH_DATA_ROOT}"
CMR_VALID_WINDOW_INDEX = f"{DATA_ROOT}/indices/cmr_valid_windows_c24_stride8_reject_any_v2.npz"
# Default: the exclusion list shipped with this package (absolute, as training may change the working directory).
_SHIPPED_EXCLUSIONS = Path(__file__).resolve().parents[1] / "data" / "exclusions" / "openh_h16_exclusions.json"
EXCLUSION_OVERLAY = "${oc.env:OPENH_EXCLUSIONS," + str(_SHIPPED_EXCLUSIONS) + "}"


@dataclass(frozen=True)
class Root:
    path: str
    # Caption context (key of ``EMBODIMENT_CAPTION_CONTEXT``) for dVRK roots.
    caption: str | None = None
    public: bool = True


@dataclass(frozen=True)
class Family:
    name: str
    ratio: float
    cls: Any
    roots: tuple[Root, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


# Settings shared by every family (H16 task-prompt recipe).
COMMON_KWARGS: dict[str, Any] = dict(
    action_normalization="quantile",
    chunk_length=24,
    enable_fast_init=False,
    fps=10.0,
    # [forward dynamics, inverse dynamics, policy, cross-view inpainting]
    joint_mode_weights=[0.4, 0.2, 0.4, 0.0],
    mode="joint",
    pose_convention="backward_anchored_world_trans",
    split="train",
    split_seed=42,
    split_val_ratio=0.05,
    viewpoint="ego_view",
    sample_stride=8,
    video_source="clean",
    exclude_episodes_overlay=EXCLUSION_OVERLAY,
    num_history_actions=16,
)

_CMR_KWARGS: dict[str, Any] = dict(
    prompt_source="task",
    video_key="observation.images.endoscope",
    letterbox_hw=[480, 832],
    valid_window_index_path=CMR_VALID_WINDOW_INDEX,
    clutch_policy="reject_any",
)


def _dvrk(embodiment: str, normalizer: str, video_key: str, **kwargs: Any) -> dict[str, Any]:
    return dict(
        normalizer_filename=normalizer,
        prompt_source=kwargs.pop("prompt_source", "task"),
        video_key=video_key,
        embodiment_type=embodiment,
        quaternion_order=kwargs.pop("quaternion_order", "xyzw"),
        **kwargs,
    )


FAMILIES: tuple[Family, ...] = (
    Family(
        "cmr_clinical",
        1.59626,
        CmrClinicalHistoryDataset,
        tuple(Root(f"cmr_surgical/{p}") for p in ("cholecystectomy", "hysterectomy", "inguinal_hernia", "prostatectomy")),
        dict(_CMR_KWARGS, normalizer_filename="cmr_clinical_all4_backward_anchored_worldtrans_rot6d.json"),
    ),
    Family(
        "dvrk_obuda",
        0.6949,
        GeneralizedDvrkHistoryDataset,
        tuple(
            Root(f"obuda/{p}", "dvrk_obuda")
            for p in (
                "pork_1", "frs_dome_1", "needlethreading_1", "needlethreading_2", "pegtransfer_1", "pegtransfer_2",
                "rollercoaster_1", "seaspike_1", "seaspike_2", "seaspike_3", "skinphantom_1",
            )
        ),  # fmt: skip
        _dvrk("dvrk_obuda", "dvrk_obuda_backward_anchored_worldtrans_rot6d.json", "observation.images.endoscope.left"),
    ),
    Family(
        "dvrk_ucsd",
        0.1892,
        GeneralizedDvrkHistoryDataset,
        (Root("ucsd/surgical_learning_dataset2", "dvrk_ucsd"), Root("ucsd/surgical_learning_dataset", "dvrk_ucsd")),
        _dvrk(
            "dvrk_ucsd",
            "dvrk_ucsd_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json",
            "observation.images.left",
            pose_source="state",
            quaternion_order="wxyz",
        ),
    ),
    Family(
        "dvrk_jhu",
        2.5928,
        GeneralizedDvrkHistoryDataset,
        (
            Root("jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled", "dvrk_wound_closure"),
            Root("jhu/imerse/nephfat/nephfat", "nephfat"),
            Root("jhu/imerse/srt_tissue_lift", "dvrk_tissue_lift"),
            Root("jhu/imerse/cao_cautery_combined", "cao_cautery"),
            Root("jhu/imerse/srth_porcine_chole", "srth_porcine_chole"),
            Root("jhu/suturebot_canon/suturebot_tissue_2", "suturebot", public=False),
            Root("jhu/suturebot_canon/suturing_jesse", "suturebot", public=False),
            Root("jhu/suturebot_canon/jesse_pickup", "suturebot", public=False),
            Root("jhu/lcsr/arcade/cholecystectomy", "dvrk_arcade"),
        ),
        # The wound-closure root was trained on clean video (without the burned-in needle
        # entry/exit dots), from a twin stream that is not public. ``video_source="clean"``
        # uses that stream when present; the public videos are used otherwise.
        _dvrk(
            "suturebot",
            "dvrk_jhu_backward_anchored_worldtrans_rot6d.json",
            "observation.images.endoscope.left",
            duplicate_frame_max_fraction=0.5,
        ),
    ),
    Family(
        "dvrk_suturebot_mono",
        0.28775,
        GeneralizedDvrkHistoryDataset,
        (Root("jhu/hf_suturebot", "suturebot", public=False),),
        _dvrk(
            "suturebot",
            "suturebot_backward_anchored_worldtrans_rot6d.json",
            "observation.images.endoscope_left",
            duplicate_frame_max_fraction=0.5,
        ),
    ),
    Family(
        "dvrk_stanford",
        0.5253,
        GeneralizedDvrkHistoryDataset,
        tuple(
            Root(f"stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/{p}", "dvrk_stanford")
            for p in ("peg_transfer", "needle_transfer", "tissue_retraction")
        ),
        _dvrk(
            "dvrk_stanford",
            "dvrk_stanford_backward_anchored_worldtrans_rot6d_shardfixed_c24_20260908.json",
            "observation.images.camera_left",
        ),
    ),
    Family(
        "dvrk_rob_surgical",
        0.6031,
        GeneralizedDvrkHistoryDataset,
        (Root("rob_surgical/all_merged_data", "rob_surgical"),),
        _dvrk(
            "rob_surgical",
            "dvrk_rob_surgical_backward_anchored_worldtrans_rot6d.json",
            "observation.images.endoscope",
            camera_motion_arm="lap",
            camera_motion_threshold=5.0,
            pose_source="action",
            prompt_source="instruction.text",
        ),
    ),
    Family(
        "dvrk_ustc",
        0.30749,
        GeneralizedDvrkHistoryDataset,
        tuple(
            Root(f"ustc/{p}", "ustc_torin", public=False)
            for p in (
                "needle_pickup", "Needle_handover", "ustc_knot_tying", "grasp_on_liver", "exvivo_liver_sep",
                "invivo_liver_sep", "tissue_lifting",
            )
        ),  # fmt: skip
        _dvrk("ustc_torin", "dvrk_ustc_backward_anchored_worldtrans_rot6d.json", "observation.images.endoscope.left"),
    ),
    Family(
        "dvrk_hamlyn",
        0.3272,
        GeneralizedDvrkHistoryDataset,
        tuple(
            Root(f"hamlyn/{p}", "hamlyn")
            for p in (
                "knot_tying", "needle_grasp_and_handover", "peg_transfer", "suturing_1", "suturing_2",
                "tissue_retraction",
            )
        ),  # fmt: skip
        _dvrk(
            "hamlyn",
            "dvrk_hamlyn_backward_anchored_worldtrans_rot6d_wxyz_c24_20260910.json",
            "observation.images.color",
            pose_source="state",
            quaternion_order="wxyz",
        ),
    ),
    Family(
        "dvrk_ucberkeley",
        0.1333,
        GeneralizedDvrkHistoryDataset,
        (Root("ucberkeley/debridement_lerobot", "dvrk_ucberkeley"),),
        _dvrk("dvrk_ucberkeley", "dvrk_ucberkeley_backward_anchored_worldtrans_rot6d.json", "observation.images.left"),
    ),
    Family(
        "turin_mitic",
        0.56609,
        GeneralizedDvrkHistoryDataset,
        tuple(
            Root(f"turin/{p}", "turin_mitic")
            for p in (
                "mitic_lerobot_plastic_tube", "mitic_lerobot_ex_vivo", "mitic_lerobot_plastic_pad",
                "mitic_lerobot_plastic_pad_3dmed",
            )
        ),  # fmt: skip
        _dvrk(
            "turin_mitic",
            "turin_mitic_backward_anchored_worldtrans_rot6d.json",
            "observation.images.endoscope.left",
            pose_source="action",
            prompt_source="instruction.text",
        ),
    ),
    Family(
        "cmr_versius_drybox_peg_transfer",
        0.00374,
        CmrClinicalHistoryDataset,
        (Root("cmr_surgical/drybox_peg_transfer", public=False),),
        dict(_CMR_KWARGS, normalizer_filename="drybox_backward_anchored_worldtrans_rot6d.json"),
    ),
    Family(
        "dvrk_star_il",
        0.0704,
        GeneralizedDvrkHistoryDataset,
        (Root("jhu/star_il_261", "star_il", public=False),),
        _dvrk(
            "star_il",
            "star_il_backward_anchored_worldtrans_rot6d.json",
            "observation.images.endoscope.left",
            pose_source="action",
        ),
    ),
    Family(
        "dvrk_tud_grasping_retraction",
        0.0287,
        GeneralizedDvrkHistoryDataset,
        (Root("tud/260131_tundra_dataset/grasping_retraction", "tud_tundra"),),
        _dvrk(
            "tud_tundra",
            "dvrk_tud_grasping_retraction_backward_anchored_worldtrans_rot6d.json",
            "observation.images.laparoscope_left",
        ),
    ),
    Family(
        "dvrk_smarts",
        0.0619,
        GeneralizedDvrkHistoryDataset,
        tuple(Root(f"jhu/lcsr/smarts/SurgSync-stitch-coldcut/{p}", "smarts") for p in ("P1", "P2", "P3")),
        _dvrk("smarts", "dvrk_smarts_backward_anchored_worldtrans_rot6d.json", "observation.images.endoscope_left"),
    ),
)

# Caption contexts the released checkpoint saw where they differ from the per-root ones
# above (only applies with ``caption_prefix_mode="legacy"``): the TUD root was resolved
# from a path substring to the USTC context.
LEGACY_CAPTION_OVERRIDES: dict[str, str] = {"tud/260131_tundra_dataset/grasping_retraction": "ustc_torin"}


def family_names(include_nonpublic: bool = False) -> list[str]:
    return [f.name for f in FAMILIES if include_nonpublic or any(r.public for r in f.roots)]


def build_dataset_entries(
    *,
    include_nonpublic: bool = False,
    caption_prefix_mode: str = "fixed",
    resolution: str = "256",
    families: list[str] | None = None,
    **dataset_overrides: Any,
) -> list:
    """``dataset_entry`` lazy configs for every family of the mixture.

    Args:
        include_nonpublic: Keep the roots that are not on the Hugging Face dataset
            (the exact training mixture of the released checkpoint).
        caption_prefix_mode: ``"fixed"`` (per-root caption context) or ``"legacy"``
            (the caption contexts the released checkpoint was trained with).
        resolution: Resolution tier of every family.
        families: Only these families (default: all).
        dataset_overrides: Extra keyword arguments for every dataset (e.g. ``max_windows``).
    """
    unknown = set(families or []) - {f.name for f in FAMILIES}
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}")
    entries = []
    for family in FAMILIES:
        if families is not None and family.name not in families:
            continue
        roots = [r for r in family.roots if include_nonpublic or r.public]
        if not roots:
            continue
        kwargs = dict(COMMON_KWARGS, **family.kwargs, **dataset_overrides)
        kwargs["roots"] = [f"{DATA_ROOT}/{r.path}" for r in roots]
        kwargs["caption_prefix_mode"] = caption_prefix_mode
        if family.cls is GeneralizedDvrkHistoryDataset:
            overrides = LEGACY_CAPTION_OVERRIDES if caption_prefix_mode == "legacy" else {}
            kwargs["caption_embodiments"] = [overrides.get(r.path, r.caption) for r in roots]
        entries.append(
            L(dataset_entry)(name=family.name, ratio=family.ratio, resolution=resolution, dataset=L(family.cls)(**kwargs))
        )
    return entries
