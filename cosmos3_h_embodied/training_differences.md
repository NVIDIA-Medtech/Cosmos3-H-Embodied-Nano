# Differences from the training run of the released checkpoint

The released checkpoint was trained with NVIDIA-internal code on NVIDIA-internal copies of the Open-H data. This
repository re-implements that training on cosmos-framework and rebuilds the data from the public Hugging Face dataset.
This page lists everything that differs between the two, and what was checked to be identical. Details and dates are
in [`design_and_decisions_log.md`](../design_and_decisions_log.md).

In short: for the public part of the data, this repository reproduces the training inputs exactly (with one
approximately re-encoded dataset) and the training computation up to floating-point precision. What cannot be
reproduced is the non-public part of the mixture, and the run's exact state (optimizer state, world size).

## 1. Data that is not public

These inputs of the training run are not on the Hugging Face dataset. The `openh_h16_task_prompt` experiment leaves
them out; `openh_h16_task_prompt_reference` lists them (`public=False` in `configs/openh_datasets.py`) and needs
them to run.

| Input | Used for | Effect without it |
|---|---|---|
| SutureBot mono root | family `dvrk_suturebot_mono` | family left out |
| USTC, 7 roots | family `dvrk_ustc` | family left out |
| CMR dry-box peg transfer | family `cmr_versius_drybox_peg_transfer` | family left out |
| STAR-IL, a version that differs from the STAR-IL data on Hugging Face | family `dvrk_star_il` | family left out |
| SutureBot "canonical" roots ×3 | 3 of the 9 roots of family `dvrk_jhu` | `dvrk_jhu` trains on the other 6 |
| CMR clinical per-frame instruction texts | captions of family `cmr_clinical` | captions fall back to the procedure description |
| Clean video of the JHU wound-closure root (without the burned-in needle entry/exit dots) | video of that root | the public video with the dots is used |

The four missing families had about 8% of the sampling weight. The remaining families keep their ratios, so their
shares grow accordingly. 11 families remain.

## 2. Data that is rebuilt from Hugging Face

`tools/data_prep` rebuilds the other 42 roots from the Hugging Face dataset (LeRobot v2.1 → v3.0, then per-root
repairs). The repairs re-implement fixes that were applied to the training copies, several of them by hand at the
time: timestamps, row indices, video windows, column layouts, `modality.json` files, the SMARTS task table, the UC
Berkeley column remap, and the CMR valid-window index.

Compared with the training copies:

- **Equal:** every data column of every root; decoded video frames of every root except Turin; the episode exclusion
  list (302 episodes); the CMR valid-window index.
- **Re-encoded video:** CAO, SRT-H, ARCADE, rob_surgical and TUD were re-encoded to H.264 for training. With
  ffmpeg 6.1.1 (Ubuntu 24.04 packages), `reencode_videos.py` reproduces them frame-exactly. Other ffmpeg/x264 builds
  may give codec-noise-level differences.
- **Turin (4 roots)** was downscaled to 854x480 for training. The rebuild matches it to 39–41 dB PSNR, not exactly.
- **Action statistics** (`data/normalizers/`) are shipped as the training run used them.
  `tools/compute_action_normalizer.py` reproduces the most recent ones (Hamlyn, UCSD) exactly. The others were
  computed earlier with a simpler procedure on an earlier state of the data; recomputed, their 1st/99th percentiles
  differ by up to ~30% on some dimensions.
- **Episode statistics** in `meta/` (not read by the loaders) differ for a few roots: some training copies carry stale
  statistics from before a repair.

## 3. Recipe and configuration

| Topic | Training run | This repository |
|---|---|---|
| Caption prefix | Selected through a lookup bug: all `dvrk_jhu` samples got the wound-closure description, all `cmr_clinical` samples the cholecystectomy description, and the TUD root the USTC description. | Per-root descriptions by default (`caption_prefix_mode: fixed`); `legacy` reproduces the training run's prefixes (used by the reference experiment). |
| Family mixing | Rank-based on 32 GPUs: each rank serves one family, ranks allocated by ratio. The effective mixture depends on the number of ranks. | Same, plus per-sample mixing (`shard_across_workers: false`) for fewer ranks than families. On a different number of ranks the effective mixture differs. |
| Starting point | The H16 training started from an earlier Open-H post-training stage of Cosmos3-Nano (neither the checkpoint nor its recipe is released). | Fine-tuning starts from the released H16 checkpoint. |
| Optimizer and schedule | Optimizer and LR-scheduler state carried through the run; the LR ended at 1e-5. | The released checkpoint has no optimizer state: a fine-tuning run starts a new optimizer. Default schedule: constant 1e-5 after 100 warm-up steps; the reference experiment restarts the original schedule. |
| Released weights | The run kept EMA weights alongside the network weights. | The released file holds the regular network weights (bf16 in the run, exported losslessly). |

## 4. Numerical differences

cosmos-framework differs from the internal training code in a few precision choices. None changes what is computed.

- Clean action targets stay in fp32 for the flow interpolation (the training code cast them to bf16 first).
- At inference, the conditioning video is normalized in fp32 (the training code: bf16; at most one bf16 step apart).
- Kernels differ (attention backend, grouped matrix products for the action projections), so results agree to
  bf16-level precision, not bit for bit.

## 5. What was validated

Run on the original data, with the original training code and this repository side by side, from the released
checkpoint:

- Dataset samples (video, actions with history, captions, sequence plans) are identical for every family.
- The rebuilt public data equals the training copies as described in section 2.
- One training step: loss agrees to 0.08% (relative, mean over 16 micro-batches). The first 20 training iterations:
  loss 0.743 vs 0.746. Resuming from a checkpoint continues every dataloader worker at the next sample.
- Evaluation on 14 families × 64 validation samples × 3 modes: action MSE agrees within statistical noise in every
  family and mode; video PSNR within 0.08 dB.
- Policy server: actions agree to 0.4% (relative RMS).

The full results are in section 7 of the design log.
