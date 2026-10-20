# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "fractal": 20,
    "dvrk_wound_closure": 21,
    "suturebot": 22,
    "srth_porcine_chole": 23,
    "cao_cautery": 24,
    "needle_pickup": 25,  # CMR Versius, Open-H cmr-surgical-60hz (CMR clinical reuses this)
    "versius_suturing": 26,  # CMR Versius suturing (same arm as needle_pickup, distinct task)
    # Open-H groups (generalized dVRK loader + bespoke). Assigned to the FREE
    # slots inside [0, num_embodiment_domains=32) so no model/checkpoint change is
    # needed: these indices exist in the base's 32-wide DomainAwareLinear but were
    # unused in pretraining (xavier-init), so they train fresh per new embodiment.
    # Bumping num_embodiment_domains would break loading the base's 32-wide action heads.
    "dvrk_arcade": 10,  # jhu/lcsr/arcade
    "dvrk_obuda": 11,  # obuda
    "dvrk_stanford": 14,  # stanford real_robot_dvrk
    "hamlyn": 16,  # hamlyn_30hz
    "dvrk_ucsd": 17,  # ucsd
    "dvrk_ucberkeley": 18,  # ucberkeley
    "turin_mitic": 19,  # turin ex-vivo
    "rob_surgical": 27,  # rob_surgical (4-arm)
    "tud_tundra": 28,  # tud TUNDRA (single-arm)
    "star_il": 29,  # jhu/imerse/star_il
    "smarts": 30,  # jhu/lcsr/smarts
    "ustc_torin": 31,  # USTC
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


# Per-embodiment caption context, distilled from the Open-H dataset READMEs
# (robot + setting + task + typical instruments). Prepended to each dVRK sample's
# caption so the model gets the "what robot / what procedure / what tools" context
# the raw task strings lack. Keyed by embodiment_type; CMR is intentionally absent
# (its grafted per-frame instruction.text_with_state labels are already rich).
EMBODIMENT_CAPTION_CONTEXT: dict[str, str] = {
    "dvrk_obuda": "da Vinci Research Kit (dVRK) dual-arm, phantom / ex-vivo surgical training with PSM graspers and needle driver.",
    "dvrk_arcade": "da Vinci Research Kit (dVRK) dual-arm on ex-vivo tissue, using a large needle driver / Cadiere forceps and monopolar cautery.",
    "hamlyn": "bimanual da Vinci Research Kit (dVRK) on ex-vivo porcine tissue and table-top rigs, with DeBakey forceps (left) and a needle driver (right).",
    "dvrk_ucsd": "da Vinci Research Kit (dVRK) dual-arm phantom retraction and tissue dissection, using bipolar forceps and Potts scissors.",
    "dvrk_ucberkeley": "da Vinci Research Kit (dVRK) dual-arm ex-vivo surgical debridement with PSM graspers.",
    "turin_mitic": "da Vinci Research Kit (dVRK) dual-arm suturing subtasks on ex-vivo porcine tissue and phantoms, with dVRK needle-handling instruments.",
    "dvrk_stanford": "da Vinci Research Kit (dVRK) dual-arm phantom needle transfer, peg transfer and tissue retraction with PSM graspers.",
    "ustc_torin": "da Vinci Research Kit (dVRK) dual-arm ex-vivo and in-vivo liver tasks, needle handling and knot tying with PSM graspers.",
    "rob_surgical": "da Vinci-style multi-arm surgical robot performing surgical manipulation with per-arm graspers.",
    "smarts": "da Vinci-Si (dVRK-Si) dual-arm ex-vivo and phantom multitask surgery (suturing, cold-cut dissection, peg transfer).",
    "star_il": "STAR (Smart Tissue Autonomous Robot) surgical system performing ex-vivo anastomosis suturing.",
    "tud_tundra": "Universal Robots UR5e surgical assistant with a stereo laparoscope, in-vivo porcine laparoscopy (grasping, retraction).",
    "suturebot": "da Vinci-Si (dVRK-Si) dual-arm table-top suturing: needle pickup, handover and knot tying with needle drivers.",
    "cao_cautery": "da Vinci-Si (dVRK-Si) dual-arm ex-vivo central-airway-obstruction tumor resection, using a grasper for retraction and a cautery hook for resection.",
    "srth_porcine_chole": "da Vinci-Si (dVRK-Si) dual-arm ex-vivo porcine cholecystectomy (clip-and-cut), using a medium-large clip applier and curved scissors.",
    "dvrk_wound_closure": "da Vinci-Si (dVRK-Si) dual-arm ex-vivo suture-pad wound closure: needle passing, knot tying and needle pickup with needle drivers.",
    "nephfat": "da Vinci-Si (dVRK-Si) dual-arm ex-vivo perinephric fat dissection and tissue mobilization with PSM graspers.",
    "dvrk_tissue_lift": "da Vinci-Si (dVRK-Si) dual-arm ex-vivo tissue lifting and retraction with PSM graspers.",
}


# Combined dVRK entries (e.g. dvrk_jhu) mix genuinely different procedures under ONE
# config embodiment_type, so a single caption prefix would mislabel every root with the
# last one's procedure. Resolve each shard root to its OWN caption context by matching a
# substring of the root path (ordered most-specific first; first hit wins). Roots with no
# match fall back to the entry's embodiment_type.
_ROOT_EMBODIMENT_PATTERNS: list[tuple[str, str]] = [
    ("wound_closure", "dvrk_wound_closure"),
    ("cao_cautery", "cao_cautery"),
    ("srth_porcine_chole", "srth_porcine_chole"),
    ("srth", "srth_porcine_chole"),
    ("nephfat", "nephfat"),
    ("tissue_lift", "dvrk_tissue_lift"),
    ("arcade", "dvrk_arcade"),
    ("obuda", "dvrk_obuda"),
    ("suturebot", "suturebot"),
    ("jesse_pickup", "suturebot"),
    ("srt_needle", "suturebot"),
    ("hamlyn", "hamlyn"),
    ("stanford", "dvrk_stanford"),
    ("ucsd", "dvrk_ucsd"),
    ("ucberkeley", "dvrk_ucberkeley"),
    ("turin", "turin_mitic"),
    ("smarts", "smarts"),
    ("star_il", "star_il"),
    ("ustc", "ustc_torin"),
    ("tundra", "tud_tundra"),
    ("rob_surgical", "rob_surgical"),
]


def resolve_root_embodiment(root: str, default: str | None = None) -> str | None:
    """Embodiment key for a shard root, by matching a substring of its path.

    First honors the entry's own ``default`` embodiment: if the root path contains the
    default's core token (e.g. ``hamlyn`` in ``.../hamlyn/Suturing-1``), keep it — so a
    single-embodiment entry is never hijacked by a generic task word in a leaf name.
    Otherwise resolve the per-root procedure (for combined entries like dvrk_jhu that mix
    procedures under one embodiment_type), falling back to ``default`` when nothing matches.
    """
    p = str(root).lower()
    if default and default.lower().replace("dvrk_", "") in p:
        return default
    for sub, emb in _ROOT_EMBODIMENT_PATTERNS:
        if sub in p:
            return emb
    return default


def caption_prefix_for_root(root: str, default_embodiment: str | None) -> str | None:
    """Per-root caption context: resolve the root's own embodiment, else the entry default."""
    emb = resolve_root_embodiment(root, default_embodiment)
    return default_caption_prefix(emb) or default_caption_prefix(default_embodiment)


def default_caption_prefix(embodiment_type: str | None) -> str | None:
    """Caption context string for an embodiment, or None if none is registered."""
    if not embodiment_type:
        return None
    return EMBODIMENT_CAPTION_CONTEXT.get(embodiment_type.lower().strip())


def cmr_caption_prefix(procedure: str | None) -> str:
    """Procedure-aware caption context for CMR Versius. The CMR entries share one
    embodiment (needle_pickup) but are distinct procedures, so the prefix is built
    from the loader's ``self._procedure`` rather than the embodiment table."""
    p = (procedure or "").lower().strip()
    if "drybox" in p or "dry box" in p or "peg" in p:
        return (
            "CMR Versius surgical robot, dry-lab training on a peg-transfer board, "
            "with articulated Versius graspers."
        )
    proc = (procedure or "surgical procedure").strip()
    return (
        f"CMR Versius surgical robot, clinical in-vivo minimally-invasive {proc}, "
        "with articulated Versius instruments (graspers, scissors, clip appliers, electrosurgery)."
    )
