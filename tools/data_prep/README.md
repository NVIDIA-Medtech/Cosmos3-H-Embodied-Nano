# Open-H data preparation

These scripts rebuild, from the public Hugging Face dataset
[`nvidia/PhysicalAI-Robotics-Open-H-Embodiment`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
(LeRobot v2.1, under `Surgical/`), the LeRobot v3.0 roots that the released H16 checkpoint was trained on.
Root paths below are relative to `Surgical/`. Run everything with the cosmos-framework Python environment, which
pins the lerobot revision used to build the training data.

## Steps

| Code | Step | Script |
|---|---|---|
| C | v2.1 → v3.0 conversion, out of place. Videos are stream-copied, so decoded frames equal the source. Absolute timestamps are rewritten to episode-relative `frame_index / fps` (float64, as in training). `modality.json`, `episodes.*_vlm_qwen3.jsonl` and `annotations.jsonl` are carried over. | `convert_v21_to_v30.py` |
| BF | Write the missing `meta/episodes_stats.jsonl` (`--mode exact` or `--mode synthesize`, as listed per root). This affects metadata only. | `episodes_stats.py` |
| VC | Video-complete shim: drops episodes without a video for every view and renumbers them 0..K-1. Episode IDs, and therefore the train/val split, change. It also removes `info.json` `splits`, so the outcome splits (failure episodes kept out of the policy mode) do not apply to these roots, as in training. | `video_complete_shim.py` |
| T | Timestamp repair, out of place: `timestamp = frame_index / fps` with timestamp stats recomputed. `--verify-root` checks the result on values. | `repair_timestamps.py` |
| I | Row-index repair, out of place: `index` = physical row position. | `repair_row_indices.py` |
| VW | Video-window fix: clips `to_timestamp` of episodes whose video window overshoots the kinematics length. Runs in place on the episode metadata and is idempotent. | `fix_video_window.py` |
| M | Install an authored `meta/modality.json` (shipped under `modality/`). | `install_modality.py` |
| D | Drop per-frame columns the loaders cannot use (string tool names, `text` dtype) from the data files, `info.json` and the episode statistics. In place, idempotent. | `drop_columns.py` |
| P | Rewrite `task_index` to row positions in `tasks.parquet` (LeRobot looks tasks up by position). In place, idempotent. | `smarts_fix_task_index_positional.py --root <parent of the leaves>` |
| S1–S6 | SMARTS chain (see below). | `smarts_*.py` |
| X | Episode exclusions: the overlay file given to the loaders (`exclude_episodes_overlay`), keyed by root path. This is not a data rewrite. | — |
| W | CMR valid-window index (`.npz`) given to the CMR loaders (`valid_window_index_path`). The recipe expects it at `<data root>/indices/cmr_valid_windows_c24_stride8_reject_any_v2.npz`; build it over the CMR roots with the defaults (fps 10, chunk 24, stride 8, threshold 5000, `reject_any`) and `PYTHONPATH=<repository root>`. | `build_cmr_valid_window_index.py` |
| E | Re-encode the training view to H.264 (`libx264 -preset fast -crf 20`, default GOP), in place, frame count checked. Settings from the training videos' x264 headers. | `reencode_videos.py --view <key>` |
| DS | Downscale + re-encode: `--scale 854x480 --crf 18` (Lanczos). | `reencode_videos.py --view <key> --scale 854x480 --crf 18` |

Run conversions in place of the upstream layout: for example, `--src-root <hf>/Surgical --dst-root <data>`
mirrors the relative paths. The loaders and the exclusion overlay identify roots by these relative paths.

**Equivalence.** Parquet values are identical for C, T, I, VW and the SMARTS chain, but the bytes differ (writer
version and compression). Stream-copied video decodes to identical frames. Re-encoded roots (E, DS) depend on the
ffmpeg/x264 build: with ffmpeg 6.1.1 (Ubuntu 24.04 packages), E reproduced the training videos frame-exactly; DS
(Turin) differs at codec-noise level (about 40 dB PSNR). Other builds differ at codec-noise level.

## Per-root chains (training roots, in order)

Steps marked ✱ are approximate (see the notes of the root).

### cmr_clinical

| Root | Chain |
|---|---|
| `cmr_surgical/{cholecystectomy, hysterectomy, inguinal_hernia, prostatectomy}` | C → W. X on `inguinal_hernia` (2 frozen episodes). |

The per-frame instruction texts used for training captions (`instruction.text_with_state`,
`meta/instruction_segments.json`) are **not public**. Without them the loader falls back to the procedure caption.

### dvrk_obuda

| Root | Chain |
|---|---|
| `obuda/{pork_1, frs_dome_1, needlethreading_1, needlethreading_2, pegtransfer_1, pegtransfer_2, rollercoaster_1, seaspike_2, seaspike_3, skinphantom_1}` | C only. Timestamp jitter of ≤1.5 ms is kept; do not repair it. |
| `obuda/seaspike_1` | C → T |

### dvrk_ucsd

| Root | Chain |
|---|---|
| `ucsd/surgical_learning_dataset2`, `ucsd/surgical_learning_dataset` | C only |

### dvrk_jhu

| Root | Chain |
|---|---|
| `jhu/imerse/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled` | C → clean-video twin (**not public**) → X (3 misaligned episodes) |
| `jhu/imerse/nephfat/nephfat` | C → T → VW → X (7 non-monotonic-timestamp episodes) |
| `jhu/imerse/srt_tissue_lift` | C only |
| `jhu/imerse/cao_cautery_combined` | C → E (`observation.images.endoscope.left`) → T (`--allow-relative`: C already rewrote the absolute timestamps; T recomputes the timestamp statistics, as in training) |
| `jhu/imerse/srth_porcine_chole` | BF exact → C → E (`observation.images.endoscope.left`). C rewrites the absolute timestamps to episode-relative values, as in training (check with `repair_timestamps.py --verify-root`); no T. |
| `jhu/lcsr/arcade/cholecystectomy` | C → M (authored 16-D quat-7 modality; none on HF) → E (`observation.images.endoscope.left`) |
| SutureBot "canonical" roots `suturebot_tissue_2`, `suturing_jesse`, `jesse_pickup` | **not public** |

### dvrk_suturebot_mono

**Not public.** The closest public root is `jhu/imerse/suturebot`, which has a different episode count and video key.

### dvrk_stanford

| Root | Chain |
|---|---|
| `stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer` | C → SF → T → I → X (10 frozen) |
| `.../needle_transfer` | C → SF → T → X (89) |
| `.../tissue_retraction` | C → SF → T → X (190) |

SF = D (`observation.meta.tool_type`, dtype `text`, which LeRobot cannot load) and P (each leaf keeps a global
`task_index`, e.g. 2 for peg_transfer). The HF `modality.json` (`pos3 + euler_xyz`) is the one used in training.

```bash
S=<data>/stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk
python tools/data_prep/drop_columns.py --root $S/peg_transfer --root $S/needle_transfer --root $S/tissue_retraction observation.meta.tool_type
python tools/data_prep/smarts_fix_task_index_positional.py --root $S
```

### dvrk_rob_surgical

| Root | Chain |
|---|---|
| `rob_surgical/all_merged_data` | BF synthesize → C → D (`observation.meta.{left,right,aux}_tool`) → E (`observation.images.endoscope`) ✱. `instruction.text` is kept. The training video has a keyframe every 26 frames; E uses x264's default GOP, which changes only seek cost, not decoded frames beyond codec noise. |

### dvrk_ustc

7 roots, **not public**.

### dvrk_hamlyn

| Root | Chain |
|---|---|
| `hamlyn/{knot_tying, needle_grasp_and_handover, peg_transfer, suturing_1, suturing_2, tissue_retraction}` | C → HM |

HM = M (the HF files point the `right_arm` block at the left arm's columns; the shipped files use pose `8:15` and
gripper `15:16` of the 16-D `observation.state`) and D (`observation.meta.{left,right}_arm_tool`).

### dvrk_ucberkeley

| Root | Chain |
|---|---|
| `ucberkeley/debridement_lerobot` | C → UB → M (video stays 1280x960) |

UB (`ucberkeley_remap.py`) makes these changes:
- sets `action ← action.cartesian_state` and `observation.state ← observation.cartesian_state`;
- drops the `observation.images.left/right` string columns and `next.done`;
- sets the features to float32 [16];
- adds a global `index`.

M then installs the quat-7 modality over the remapped columns.

### turin_mitic

| Root | Chain | Training totals (episodes / frames) |
|---|---|---|
| `turin/mitic_lerobot_plastic_tube` | C → DS (`observation.images.endoscope.left`) | 480 / 216,070 |
| `turin/mitic_lerobot_ex_vivo` | VC → BF exact → C → modality copy → DS | 792 / 387,204 |
| `turin/mitic_lerobot_plastic_pad` | same as `ex_vivo` | 544 / 149,070 |
| `turin/mitic_lerobot_plastic_pad_3dmed` | same as `ex_vivo` | 368 / 242,946 |

The training videos are H.264 854x480 (CRF 18), although `info.json` still describes the source AV1 1920x1080. DS encodes from the AV1 source in one pass; whether training had an intermediate full-size H.264 pass cannot be told from the final file.

### cmr_versius_drybox_peg_transfer

**Not public.** It is a 360p re-segmentation; no public root matches. HF `cmr_surgical/dry_box` also carries an
undeclared `instruction.text_with_state` column and does not load as converted.

### dvrk_star_il

**Not public.** The 252-episode training root differs from HF `jhu/imerse/star_il/star_il`, which has 512 episodes
and per-episode constant timestamps.

### dvrk_tud_grasping_retraction

| Root | Chain |
|---|---|
| `tud/260131_tundra_dataset/grasping_retraction` | BF exact (source stats contain NaN) → C → M (`modality/tud/260131_tundra_dataset/grasping_retraction/modality.json`; the upstream file is not valid JSON) → E (MPEG-4 → H.264, `observation.images.laparoscope_left`) |

### dvrk_smarts

| Root | Chain |
|---|---|
| `jhu/lcsr/smarts/SurgSync-stitch-coldcut/{P1, P2, P3}` | S1 → S2 → C → S3 → S4 → S5 (no sidecars) |

1. **S1** `smarts_preprocess_v21.py`: assembles 16-D action and state from the scalar pose columns (pose from
   `observation.cartesian_state`), renumbers the episodes 0..N-1, writes `index`, symlinks the videos and writes a
   quat-7 modality.
2. **S2** `episodes_stats.py --mode synthesize`.
3. **S3** `smarts_patch_scalar_columns.py` and **S4** `smarts_patch_tasks_column.py`: fixes needed by data produced
   with an earlier version of S1. They are idempotent no-ops on S1 output and are kept so the chain matches training.
4. **S5** `smarts_fix_task_index_positional.py`: rewrites `task_index` to row positions in `tasks.parquet`.
   **Required**: without it, prompts are wrong.

The training run read the original 1920x1080 H.264 video (no re-encode).

## Local sample fixtures

`renumber_sample_episodes.py` makes trimmed sample roots (from `tools/download_openh_samples.py`) contiguous before
conversion. It is not part of the training-data rebuild.
