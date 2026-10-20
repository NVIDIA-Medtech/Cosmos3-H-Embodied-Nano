Frames at 176 rows; FVD on 16-frame clips every 8 frames. Difference = c3hen - chss (lower is better); 95% CI and the share of 200 paired bootstrap resamples (over clips) in which c3hen is better.

| Pool | Window | Metric | c3hen | chss | c3hen input | chss input | Difference | 95% CI | Share better |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CMR Versius, 31.2 s | all | FID | 113.7 | 90.1 | 7.6 | 1.3 | +23.6 | [+9.9, +37.2] | 0% |
| CMR Versius, 31.2 s | all | FVD | 794.2 | 382.2 | 14.1 | 4.5 | +412.0 | [+357.1, +523.3] | 0% |
| CMR Versius, 31.2 s | 0-6 s | FID | 64.4 | 71.4 | 12.7 | 2.6 | -7.0 | [-12.6, +0.9] | 96% |
| CMR Versius, 31.2 s | 0-6 s | FVD | 501.2 | 428.0 | 25.7 | 7.4 | +73.3 | [+11.3, +171.7] | 2% |
| CMR Versius, 31.2 s | 6-12 s | FID | 124.5 | 112.0 | 13.3 | 2.6 | +12.4 | [-1.9, +30.1] | 7% |
| CMR Versius, 31.2 s | 6-12 s | FVD | 1103.3 | 647.6 | 24.9 | 6.7 | +455.7 | [+357.8, +668.4] | 0% |
| CMR Versius, 31.2 s | 12-18 s | FID | 155.1 | 127.8 | 13.3 | 2.6 | +27.3 | [+7.3, +47.2] | 0% |
| CMR Versius, 31.2 s | 12-18 s | FVD | 1233.8 | 655.2 | 25.8 | 6.9 | +578.6 | [+451.3, +873.8] | 0% |
| CMR Versius, 31.2 s | 18-24 s | FID | 196.6 | 150.2 | 13.1 | 2.7 | +46.4 | [+26.4, +66.4] | 0% |
| CMR Versius, 31.2 s | 18-24 s | FVD | 1549.6 | 876.2 | 26.4 | 7.7 | +673.4 | [+519.5, +932.5] | 0% |
| CMR Versius, 31.2 s | 24-31 s | FID | 217.8 | 159.4 | 12.5 | 2.5 | +58.4 | [+31.6, +81.8] | 0% |
| CMR Versius, 31.2 s | 24-31 s | FVD | 1764.6 | 960.7 | 24.2 | 6.6 | +803.9 | [+621.7, +1041.4] | 0% |

Per window: pixel metrics (mean over rollouts) and ratios to the recorded video (1 is ideal):

| Pool | Window | Model | PSNR | SSIM | Dynamic PSNR | Sharpness | Motion |
|---|---|---|---:|---:|---:|---:|---:|
| CMR Versius, 31.2 s | 0-6 s | c3hen | 16.71 | 0.354 | 14.42 | 0.93 | 0.94 |
| CMR Versius, 31.2 s | 0-6 s | chss | 16.76 | 0.372 | 14.44 | 0.86 | 0.81 |
| CMR Versius, 31.2 s | 6-12 s | c3hen | 14.40 | 0.250 | 13.11 | 0.74 | 1.22 |
| CMR Versius, 31.2 s | 6-12 s | chss | 14.73 | 0.262 | 13.19 | 0.81 | 0.87 |
| CMR Versius, 31.2 s | 12-18 s | c3hen | 14.30 | 0.239 | 13.08 | 0.66 | 0.79 |
| CMR Versius, 31.2 s | 12-18 s | chss | 14.05 | 0.228 | 12.83 | 0.78 | 1.00 |
| CMR Versius, 31.2 s | 18-24 s | c3hen | 13.85 | 0.225 | 13.02 | 0.62 | 0.80 |
| CMR Versius, 31.2 s | 18-24 s | chss | 13.55 | 0.200 | 12.75 | 0.76 | 0.98 |
| CMR Versius, 31.2 s | 24-31 s | c3hen | 13.24 | 0.224 | 12.36 | 0.59 | 1.36 |
| CMR Versius, 31.2 s | 24-31 s | chss | 13.12 | 0.183 | 12.39 | 0.75 | 1.10 |

Per window, paired pixel differences (c3hen - chss):

| Pool | Window | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---|---:|---:|---:|---:|
| CMR Versius, 31.2 s | 0-6 s | psnr | -0.0510 | ±0.2591 | 50% | 0.76 |
| CMR Versius, 31.2 s | 0-6 s | ssim | -0.0180 | ±0.0107 | 27% | 0.0027 |
| CMR Versius, 31.2 s | 0-6 s | dynamic_psnr | -0.0122 | ±0.1634 | 48% | 0.94 |
| CMR Versius, 31.2 s | 6-12 s | psnr | -0.3281 | ±0.4605 | 45% | 0.54 |
| CMR Versius, 31.2 s | 6-12 s | ssim | -0.0113 | ±0.0194 | 55% | 0.74 |
| CMR Versius, 31.2 s | 6-12 s | dynamic_psnr | -0.0796 | ±0.2702 | 48% | 0.79 |
| CMR Versius, 31.2 s | 12-18 s | psnr | +0.2525 | ±0.4345 | 66% | 0.073 |
| CMR Versius, 31.2 s | 12-18 s | ssim | +0.0109 | ±0.0184 | 73% | 0.012 |
| CMR Versius, 31.2 s | 12-18 s | dynamic_psnr | +0.2516 | ±0.3654 | 61% | 0.1 |
| CMR Versius, 31.2 s | 18-24 s | psnr | +0.3019 | ±0.3928 | 66% | 0.056 |
| CMR Versius, 31.2 s | 18-24 s | ssim | +0.0256 | ±0.0175 | 80% | 0.00082 |
| CMR Versius, 31.2 s | 18-24 s | dynamic_psnr | +0.2651 | ±0.3568 | 61% | 0.067 |
| CMR Versius, 31.2 s | 24-31 s | psnr | +0.1180 | ±0.4261 | 55% | 0.46 |
| CMR Versius, 31.2 s | 24-31 s | ssim | +0.0415 | ±0.0188 | 84% | 5.8e-05 |
| CMR Versius, 31.2 s | 24-31 s | dynamic_psnr | -0.0342 | ±0.3618 | 57% | 0.6 |
