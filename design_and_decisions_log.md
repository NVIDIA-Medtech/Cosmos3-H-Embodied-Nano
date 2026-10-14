# Design and decisions log

Running record of design decisions for the Cosmos3-H-Embodied-Nano public code, and of every place where it
deliberately diverges from the internal training code ("the fork") that produced the released checkpoint.
Newest entries are appended at the bottom of each section. Dates are PT.

Terminology:
- **fork**: the internal training codebase revision that trained the task-prompt H16 50k checkpoint.
- **CF**: [cosmos-framework](https://github.com/NVIDIA/cosmos-framework), the public Cosmos3 training stack.
- **HF dataset**: [`nvidia/PhysicalAI-Robotics-Open-H-Embodiment`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
  (LeRobot v2.1).
- **H16**: up to 16 past action transitions given to the model as clean (un-noised) history.

## 1. Architecture

| Date | Decision | Rationale |
|---|---|---|
| 2026-10-12 | Implement the Open-H pipeline as a plugin package (`cosmos3_h_embodied`) on top of a pinned CF; no CF fork. | The fork's code base derives from a non-public predecessor of CF. CF already supports H16 at the model level (clean history rows, mRoPE offsets, FD/ID/WAM plans). |
| 2026-10-13 | Port the fork's data modules (datasets, history, normalization, transforms, mixing) largely verbatim, then trim. Reuse CF modules where they are verified identical to the fork's (checked symbol by symbol). | Equivalence with the training run of the released checkpoint is the primary requirement. |
| 2026-10-13 | Reuse CF's `IterativeJointDataLoader` packing loop (same `>=` length limit, lookahead 10) with a subclass that charges 40 action tokens per sample regardless of history length. | Same packing behaviour as the fork's history-matched loader, without duplicating CF code. |
| 2026-10-13 | Equivalence tests import the fork and the port side by side in the CF environment (same lerobot, torch, torchcodec, numpy, pyarrow pins). These tests are internal and are not published (decision 2026-10-14); results are summarized in this log. | Sample-level parity checks on real data. |
| 2026-10-13 | The history-matched loader computes the per-sample token count itself (fork formula) instead of using CF's. CF rounds the patch grid up (`ceil((H//16)/2)`), the fork rounds down (`H//32`); they differ only when `H//16` is odd, which none of the recipe's 256-tier shapes are. | Exact packing parity for any shape; verified by a mutation check (CF's count changes the batch sequence on an odd shape). |
| 2026-10-13 | Resume: CF's checkpointer already stores per-rank dataloader state through any callback tagged `checkpoint_component="dataloader"`. This package adds `ActionDataLoaderStateCallback` (the fork's `action_random` state: last sample index and RNG state per worker). Worker spawning is deferred until after checkpoint load with CF's `lazy_initialize_child_iterators=True` (fork: `prewarm: false` + `prepare_after_resume()`). | CF's own dataloader-state callback only implements the webdataset/Lance distributors. |
| 2026-10-13 | Recipe as a Hydra experiment registered by `cosmos3_h_embodied/configs/config.py` (CF's `make_config` + this package's experiments). CF's own tools take it as `--config-file` (`export_config`, then `_train --config-file <exported yaml>`); no CF patch, no copied entry point. Model config = CF's `NANO_MODEL_CONFIG` plus the recipe delta (fork Nano config → fork H16 config). | An (internal) test exports the reference experiment and diffs it against the fork's resolved training config; every remaining difference is on an explicit allowlist (moved keys, inert CF-only defaults, documented divergences). |
| 2026-10-13 | `fsdp_reduce_dtype: float32` and `cpu_offloading_activations: false` set explicitly. | The fork reduced gradients in fp32; CF's default for this config reduces in bf16. Found by the config diff. |
| 2026-10-13 | Released checkpoint → training: `tools/convert_checkpoint_to_dcp.py` writes the `.pt` network weights as a DCP `model` component (`net.*` and `net_ema.*`, original dtype; DCP casts to fp32 on load). CF's checkpointer does not load `.pt` files. | Verified: the model built from the public config has exactly the checkpoint's 809 tensors (names and shapes). |

## 2. Scope

| Date | Decision |
|---|---|
| 2026-10-13 | Release checkpoint = task-prompt H16 50k. Local equivalence tests use its bf16 export; an fp32 export exists. |
| 2026-10-13 | Only the H16 continuation recipe is provided. It resumes from the released checkpoint. The Open-H stage-1 ("mid-training") recipe that produced the H16 starting point is out of scope. |
| 2026-10-13 | The HF dataset will not be changed for this release, so all data fixes applied for training must be reproducible by code shipped here (see §4). |
| 2026-10-13 | The USTC family (7 roots) is part of the recipe although it is not yet on the HF dataset. No special handling in code. |
| 2026-10-14 | Default LR schedule of the public experiments: constant 1e-5 after a 100-step warm-up (the rate the released checkpoint's training ended at, factor 0.05 of lr 2e-4). The reference experiment keeps the schedule as trained (warm-up, then linear 8e-5 → 1e-5 over 50,000 iterations). |
| 2026-10-14 | Release checkpoint format: bf16 network weights (an fp32 export exists). Training from it is unaffected: DCP casts bf16 exactly into the fp32 FSDP master weights; action targets stay fp32 (CF). |
| 2026-10-14 | Experiments: `openh_h16_task_prompt` (HF-only, 11 families), `openh_h16_task_prompt_reference` (55 training roots, 15 families, legacy captions, LR as trained), `openh_h16_task_prompt_smoke` (5 families, per-sample mixing, window cap; one GPU with `COSMOS_SMOKE=1`). |

## 3. Divergences from the fork (behavioural)

Each divergence is either a **default change** with a setting that restores the fork behaviour, or a
**non-behavioural** change (naming, packaging, removed internal tooling).

| Date | Area | Fork behaviour | Public default | Fork-exact setting |
|---|---|---|---|---|
| 2026-10-13 | Caption prefix | The prefix lookup used a worker-local index (always 0) against a table keyed by global root index, and resolved embodiments by substrings of absolute paths. Effects: every `dvrk_jhu` sample gets the wound-closure description; every `cmr_clinical` sample gets the cholecystectomy description (and root 0's fallback text/segments); `dvrk_tud_grasping_retraction` gets the USTC description (a path component contained "ustc"). | Per-root prefixes, configured explicitly per root (no path-substring matching). | `caption_prefix_mode: legacy` reproduces the fork's effective prefixes. Equivalence tests use it; the released checkpoint was trained with it. |
| 2026-10-13 | Family mixing | Rank-based: Hare–Niemeyer allocation of global ranks to families (≥1 rank per family); each DataLoader worker serves one shard. Depends on world size (needs ≥15 ranks). | Both offered: rank-based (`shard_across_workers: true`, fork-exact, used for equivalence) and per-sample mixing (`shard_across_workers: false`: every worker loads all families and draws a family per sample by ratio) for setups with fewer ranks than families. Both modes exist in the fork's code; only rank-based was used for training. | `shard_across_workers: true` |
| 2026-10-13 | Action loss normalization | `normalize_action_loss_by_active=true`: action loss averaged over the active (noised) action rows only. CF averages over all action rows including clean history rows, which scales the action loss by 24/40 = 0.6 at h=16. Not a CF bug, a design difference. | Implemented in this package: `HistoryOmniMoTModel(normalize_action_loss_by_active=True)` wraps CF's `_compute_losses` and switches only the action loss call to `normalize_by_active=True` (no copy of the 300-line method). An internal test guards the CF call site it relies on. On in the H16 recipe. | — (same as fork) |

## 4. Data preparation

| Date | Decision |
|---|---|
| 2026-10-13 | Ship regeneration scripts plus small data files, not repaired data. Scripts rebuild from the HF dataset what training read: LeRobot v2.1 → v3.0 conversion and the repairs (timestamps, row indices, clean-video twins, CMR valid-window index, CMR instruction texts). Hand-curated inputs (episode exclusion list, CMR instruction texts if releasable) ship as data files keyed by dataset-relative paths. |
| 2026-10-13 | Normalizer statistics files are shipped unchanged in their statistics blocks; their `metadata` blocks are rewritten to dataset-relative paths (internal absolute paths removed). Loaders read only the statistics blocks. |
| 2026-10-14 | Exclusion list shipped as `cosmos3_h_embodied/data/exclusions/openh_h16_exclusions.json`: the 302 episodes of the training overlay with reasons, keyed by HF-relative paths (verified: equal to the training overlay). |
| 2026-10-14 | CMR valid-window index is rebuilt by `tools/data_prep/build_cmr_valid_window_index.py` (not shipped). Verified: rebuilt from HF data, its valid-window runs equal the training index for all locally available episodes of the four clinical procedures. |
| 2026-10-14 | `modality.json` files come from the HF dataset (the sample downloader now fetches them), except authored/fixed ones shipped under `tools/data_prep/modality/`. HF's Hamlyn files declare overlapping left/right columns (unusable); ARCADE has none on HF; the fork's fixed files are to be collected from the training data. |
| 2026-10-14 | Training `modality.json` files collected for all 55 roots and compared with HF (semantic JSON equality). Equal: 36 public roots (incl. Stanford, whose HF file is already the authored pos3 + euler_xyz layout). Shipped from training: Hamlyn ×6 (right arm at `8:15` / `15:16`), UC Berkeley (quat-7 over the remapped Cartesian columns), ARCADE (absent on HF), TUD (same content as before). Wound closure differs only by the clean-video key of the non-public twin, so HF's file is used. |
| 2026-10-14 | Column fixes as scripts: `drop_columns.py` (Stanford `observation.meta.tool_type`; Hamlyn `observation.meta.{left,right}_arm_tool`; rob_surgical `observation.meta.{left,right,aux}_tool`), `ucberkeley_remap.py` (state/action ← Cartesian columns, drop `next.done` and the v2.1 image-path strings, add `index`), and `smarts_fix_task_index_positional.py` reused for Stanford's per-leaf global `task_index`. Feature names and dtypes of the results equal the training data's `info.json` (Stanford, UC Berkeley; checked on the cluster). Value-level comparison with the training data is part of cluster validation. |
| 2026-10-14 | Training `info.json` of all 55 roots compared with the local HF rebuilds: features (names, dtypes, shapes) are equal for every public root, except the non-public CMR `instruction.text_with_state` and wound-closure clean-twin videos. ARCADE and wound closure keep their string tool columns in training (no drop). Turin roots after VC have no `splits` in training (VC removes them; the rebuild does the same). |
| 2026-10-14 | Exclusion list verified against what training applied: per root, the union of the `meta/exclude_episodes.json` sidecar and the overlay entry with the exact root path equals the shipped list (302 episodes, 7 roots). |
| 2026-10-14 | Recipe roots mirror the HF layout below `Surgical/` (fix: SMARTS is `jhu/lcsr/smarts/...`). |

## 5. Removed internal tooling (non-behavioural)

| Date | Removed | Why |
|---|---|---|
| 2026-10-13 | Internal data-profile machinery (sha256-pinned profiles bound to cluster paths), cluster preflight checks | Replaced by explicit config values and data-prep scripts. |
| 2026-10-13 | Worker memory profiler hooks, latent-cache reader, NVDEC environment switches | Not used by the H16 recipe. |
| 2026-10-13 | Environment switches read by the data code (`COSMOS_CAMERA_OVERLAP_MAX`, `COSMOS_COARSE_DROPOUT`, `COSMOS_SUBSET_MAX_WINDOWS`, `ACTION_SMOKE_FAMILIES` + `ACTION_ALLOW_UNDERSUBSCRIBED_FAMILIES`) became constructor options (`camera_overlap_max=0.25`, `coarse_dropout=None`, `max_windows=None`, `smoke_families=None`) with the fork's defaults. `COSMOS3_FD_MONO_FORCE` and `COSMOS_CAPTION_DEBUG` (debug only) were removed; decoding is CPU-only. | Configuration lives in the experiment config, not the job environment. Parity test: fork with env vars = port with options. |
| 2026-10-13 | The fork's committed wound-closure known-bad sample list is not shipped. Its two keys (`*/dvrk_wound_closure`) match none of the recipe roots, so it never had an effect in the H16 run. The cumulative runtime bad-sample log is not read by the H16 history datasets (fork: `runtime_log=""`); writing it is now opt-in (`DVRK_BAD_SAMPLES_LOG`) instead of a fixed container path. | No behavioural effect; removes internal batch names and paths. |

## 6. Open items

- ~~Stanford peg_transfer exclusions possibly not applied in training.~~ **Resolved 2026-10-14: they were applied.**
  The overlay key does not match the training root (a row-index-repaired copy, not a symlink), but every training
  root also carries a `meta/exclude_episodes.json` sidecar, which the loader reads as well; the peg_transfer
  sidecar lists the same 10 episodes. Per root, sidecar ∪ exact-path overlay entry equals the shipped list (302).
- **Inference parity, fork vs port (2026-10-14).** Same released checkpoint, same USTC samples, seed 20260908.
  Model inputs (video, actions with history, caption, plan, domain id) are bit-identical. One sampling step without
  guidance differs by 0.8% relative RMS (bf16-level numerics: fp32 action targets, grouped-matmul action projections,
  attention backend); guidance 6 amplifies it about six-fold; after 35 steps PSNR differs by ≤0.5 dB and action MSE by a
  few percent per sample. Treated as a residual numerical difference; aggregate eval metrics on a larger sample set are the
  acceptance check. Eval semantics: CF names the policy mode `wam`; dataset batches carry no negative prompt in either
  code base (unconditional branch = empty caption).
- ~~**LR schedule when continuing from the released checkpoint (2026-10-13, open).**~~ **Resolved 2026-10-14**, see §2. The recipe copies the fork's
  scheduler (LambdaLinear, warm-up 100, cycle 50,000, f_max 0.4, f_min 0.05). The fork continued with its optimizer
  and scheduler state from iteration 47,437 (LR ≈ 1.4e-5 falling to 1e-5); the released checkpoint has no optimizer
  state, so a public run starts the schedule at iteration 0 (warm-up to 8e-5). To be decided which schedule the public
  recipe should default to.
- **Action x0 precision (G2), 2026-10-13.** The fork converted the clean action targets to the model dtype (bf16) before
  flow interpolation; CF keeps them in fp32 by design (noise and velocity target without bf16 quantization). The port
  keeps CF's fp32. This is a numerical-precision difference only (no change in semantics); model-level parity tests
  use a tolerance for it. **Decided 2026-10-14: keep fp32** (see §2).
- **Non-public training inputs (2026-10-13).** The provenance trace shows that several training roots cannot be rebuilt
  from the HF dataset because their inputs were never published:
  - CMR clinical per-frame instruction texts (captions for ~20% of the mixture);
  - wound-closure "clean" video twins (no needle dots; required by one dVRK JHU root);
  - three SutureBot "canonical" roots, the SutureBot mono root, the STAR-IL root (252 episodes, different from HF STAR-IL), the CMR dry-box peg-transfer root, and USTC.
  **Decision (2026-10-13): HF-only, documented.** The public code ships only what can be rebuilt from the public
  dataset. Documentation lists exactly which roots and captions differ from the released checkpoint's training.
  Internal equivalence tests against the fork cover every reproducible part (and run on the cluster on the fork's data).
- **Re-encoded videos.** Several roots were re-encoded (AV1/MPEG-4 → H.264, some downscaled). Re-encoding is not
  byte-reproducible; decoded frames will match only to codec-noise level. Stream-copied roots decode identically.
  2026-10-14: settings taken from the training videos' x264 headers (CAO, SRT-H, ARCADE, rob_surgical, TUD: preset
  fast, CRF 20, default GOP; Turin: Lanczos 854x480, CRF 18) and implemented in `tools/data_prep/reencode_videos.py`,
  which also checks that no frames are lost. The fork's first AV1 → H.264 pass is documented to have dropped frames
  on CAO and SRT-H (videos shorter than the kinematics; episode tails unusable); if the training copies still show
  this, the rebuilt roots have more usable windows than training had. To be checked on the cluster (frame counts).
- **Undocumented in-place fixes** (Stanford, Hamlyn, UC Berkeley, rob_surgical column changes, authored `modality.json`
  files) had no committed scripts in the fork. Re-implemented 2026-10-14 (§4); with them every local HF sample root of
  these families loads and gives samples identical to the fork's dataset code. Value-level comparison with the fork's
  data is pending (cluster).
- ~~Whether `COSMOS_COARSE_DROPOUT` was enabled in the H16 run.~~ **Resolved 2026-10-13:** not set by the launcher or
  the job environment file; loader defaults applied (coarse dropout off, camera-overlap max 0.25, no window cap). The
  port's defaults are the same. (An inherited shell variable cannot be excluded from the launch records alone.)
- **Fork-data checks, 2026-10-13** (on the fork's data, read-only):
  - Nephfat: the video-window fix was applied (episode 1504 windows clipped to 337 frames) → the data-prep chain includes it.
  - TUD: `modality.json` is authored (the HF copy is not usable) → shipped as a data file; `next.done` is on HF too.
  - rob_surgical: `instruction.text` is present on HF and in the fork data → no fix.
  - SMARTS stitch-coldcut: no camera-motion frame sidecar → none shipped.
  - Turin: videos were downscaled to H.264 854x480 (approximate re-encode step).
- **Wound-closure clean video (2026-10-13):** the public recipe keeps `video_source: clean` but does not require the
  (non-public) clean twin, so with HF data the root trains on the video with burned-in needle dots.
- **Public recipe variants (2026-10-13):** `openh_h16_task_prompt` (HF-only roots, fixed captions) and
  `openh_h16_task_prompt_reference` (all 55 training roots, legacy captions, needs the internal data; used for
  equivalence runs). Fixed captions: USTC `tissue_lifting` uses the USTC context (the fork resolved it to the JHU
  tissue-lift context by path substring; legacy mode never used that entry because of the worker-local lookup).
