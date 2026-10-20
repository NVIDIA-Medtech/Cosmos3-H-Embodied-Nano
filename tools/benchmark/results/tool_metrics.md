| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | Missed tools | Tools per frame |
|---|---:|---|---:|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 18 | reference | | | | | 2.03 |
| cmr_surgical/cholecystectomy | 18 | c3hen | 0.496 / 0.575 | 76.2 / 52.2 | 21.3 | 34% | 1.83 |
| cmr_surgical/cholecystectomy | 18 | chss | 0.430 / 0.469 | 53.7 / 47.7 | 24.4 | 21% | 2.33 |
| cmr_surgical/hysterectomy | 34 | reference | | | | | 2.52 |
| cmr_surgical/hysterectomy | 34 | c3hen | 0.580 / 0.511 | 92.4 / 77.9 | 32.9 | 38% | 2.02 |
| cmr_surgical/hysterectomy | 34 | chss | 0.548 / 0.515 | 73.1 / 62.8 | 27.3 | 31% | 2.22 |
| cmr_surgical/inguinal_hernia | 32 | reference | | | | | 1.72 |
| cmr_surgical/inguinal_hernia | 32 | c3hen | 0.501 / 0.511 | 47.5 / 32.1 | 21.6 | 22% | 1.74 |
| cmr_surgical/inguinal_hernia | 32 | chss | 0.414 / 0.385 | 64.3 / 29.1 | 21.8 | 27% | 1.52 |
| cmr_surgical/prostatectomy | 44 | reference | | | | | 2.36 |
| cmr_surgical/prostatectomy | 44 | c3hen | 0.404 / 0.427 | 78.6 / 67.4 | 37.3 | 31% | 1.94 |
| cmr_surgical/prostatectomy | 44 | chss | 0.382 / 0.386 | 68.0 / 63.7 | 35.2 | 26% | 2.07 |
| **All** | 128 | reference | | | | | 2.20 |
| **All** | 128 | c3hen | 0.488 / 0.468 | 74.1 / 65.5 | 29.9 | 31% | 1.89 |
| **All** | 128 | chss | 0.441 / 0.418 | 66.4 / 56.8 | 28.2 | 27% | 2.01 |

Resolution control: each model's own input frames scored as if they were its prediction (one per clip):

| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | Missed tools | Tools per frame |
|---|---:|---|---:|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 9 | c3hen input | 0.977 / 0.981 | 1.1 / 0.7 | 3.0 | 5% | 2.16 |
| cmr_surgical/cholecystectomy | 9 | chss input | 0.964 / 0.962 | 0.4 / 0.3 | 0.8 | 4% | 2.06 |
| cmr_surgical/hysterectomy | 17 | c3hen input | 0.980 / 0.981 | 12.0 / 1.1 | 3.6 | 13% | 2.50 |
| cmr_surgical/hysterectomy | 17 | chss input | 0.967 / 0.968 | 11.2 / 0.3 | 1.3 | 10% | 2.56 |
| cmr_surgical/inguinal_hernia | 16 | c3hen input | 0.967 / 0.967 | 0.4 / 0.4 | 0.8 | 5% | 1.70 |
| cmr_surgical/inguinal_hernia | 16 | chss input | 0.958 / 0.960 | 0.2 / 0.2 | 0.5 | 3% | 1.74 |
| cmr_surgical/prostatectomy | 22 | c3hen input | 0.974 / 0.974 | 1.1 / 0.9 | 2.0 | 7% | 2.30 |
| cmr_surgical/prostatectomy | 22 | chss input | 0.960 / 0.960 | 0.4 / 0.3 | 0.7 | 4% | 2.36 |
| **All** | 64 | c3hen input | 0.974 / 0.976 | 3.8 / 0.7 | 2.2 | 8% | 2.18 |
| **All** | 64 | chss input | 0.962 / 0.962 | 3.2 / 0.3 | 0.8 | 5% | 2.22 |

| Phase | Model | GATC ↑ | TCD px ↓ |
|---|---|---:|---:|
| early | c3hen | 0.660 | 53.0 |
| early | chss | 0.661 | 50.5 |
| mid | c3hen | 0.529 | 70.3 |
| mid | chss | 0.487 | 68.4 |
| late | c3hen | 0.398 | 83.9 |
| late | chss | 0.360 | 76.4 |

Paired differences (c3hen-chss; win rate = share of rollouts where c3hen is better):

| Dataset | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | gatc | +0.0663 | ±0.0524 | 72% | 0.009 |
| cmr_surgical/cholecystectomy | tcd | +22.4771 | ±26.2213 | 28% | 0.14 |
| cmr_surgical/hysterectomy | gatc | +0.0315 | ±0.0321 | 68% | 0.033 |
| cmr_surgical/hysterectomy | tcd | +19.2385 | ±11.8819 | 18% | 0.0017 |
| cmr_surgical/inguinal_hernia | gatc | +0.0866 | ±0.0537 | 72% | 0.0041 |
| cmr_surgical/inguinal_hernia | tcd | -16.7555 | ±20.3900 | 50% | 0.31 |
| cmr_surgical/prostatectomy | gatc | +0.0218 | ±0.0395 | 55% | 0.32 |
| cmr_surgical/prostatectomy | tcd | +10.5982 | ±8.2097 | 34% | 0.02 |
| **All** | gatc | +0.0468 | ±0.0224 | 65% | 8.5e-05 |
| **All** | tcd | +7.7253 | ±7.9064 | 33% | 0.0055 |
