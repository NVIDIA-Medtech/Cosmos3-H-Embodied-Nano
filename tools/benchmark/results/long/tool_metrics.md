| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | Missed tools | Tools per frame |
|---|---:|---|---:|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 6 | reference | | | | | 2.18 |
| cmr_surgical/cholecystectomy | 6 | c3hen | 0.242 / 0.223 | 129.5 / 132.3 | 52.3 | 48% | 1.65 |
| cmr_surgical/cholecystectomy | 6 | chss | 0.338 / 0.353 | 75.9 / 78.3 | 44.6 | 27% | 2.64 |
| cmr_surgical/hysterectomy | 6 | reference | | | | | 2.78 |
| cmr_surgical/hysterectomy | 6 | c3hen | 0.377 / 0.377 | 154.2 / 184.4 | 35.2 | 65% | 1.09 |
| cmr_surgical/hysterectomy | 6 | chss | 0.465 / 0.453 | 109.8 / 87.2 | 41.4 | 42% | 1.87 |
| cmr_surgical/inguinal_hernia | 10 | reference | | | | | 1.86 |
| cmr_surgical/inguinal_hernia | 10 | c3hen | 0.238 / 0.233 | 69.3 / 76.0 | 36.0 | 28% | 1.76 |
| cmr_surgical/inguinal_hernia | 10 | chss | 0.185 / 0.183 | 74.3 / 69.5 | 38.6 | 32% | 1.43 |
| cmr_surgical/prostatectomy | 22 | reference | | | | | 2.24 |
| cmr_surgical/prostatectomy | 22 | c3hen | 0.156 / 0.143 | 148.0 / 151.2 | 57.6 | 55% | 1.23 |
| cmr_surgical/prostatectomy | 22 | chss | 0.163 / 0.146 | 113.5 / 102.9 | 59.1 | 39% | 1.64 |
| **All** | 44 | reference | | | | | 2.22 |
| **All** | 44 | c3hen | 0.217 / 0.190 | 128.4 / 131.2 | 48.9 | 49% | 1.39 |
| **All** | 44 | chss | 0.233 / 0.186 | 99.0 / 94.0 | 50.1 | 36% | 1.76 |

Resolution control: each model's own input frames scored as if they were its prediction (one per clip):

| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | Missed tools | Tools per frame |
|---|---:|---|---:|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 3 | c3hen input | 0.977 / 0.979 | 1.1 / 1.4 | 4.4 | 8% | 2.30 |
| cmr_surgical/cholecystectomy | 3 | chss input | 0.967 / 0.967 | 0.4 / 0.3 | 1.6 | 4% | 2.18 |
| cmr_surgical/hysterectomy | 3 | c3hen input | 0.978 / 0.973 | 0.8 / 0.7 | 2.2 | 6% | 2.80 |
| cmr_surgical/hysterectomy | 3 | chss input | 0.966 / 0.966 | 0.3 / 0.2 | 1.0 | 4% | 2.79 |
| cmr_surgical/inguinal_hernia | 5 | c3hen input | 0.975 / 0.976 | 0.3 / 0.3 | 0.9 | 3% | 1.85 |
| cmr_surgical/inguinal_hernia | 5 | chss input | 0.965 / 0.964 | 0.2 / 0.2 | 0.6 | 2% | 1.86 |
| cmr_surgical/prostatectomy | 11 | c3hen input | 0.977 / 0.979 | 7.0 / 0.9 | 2.5 | 10% | 2.18 |
| cmr_surgical/prostatectomy | 11 | chss input | 0.962 / 0.963 | 0.3 / 0.3 | 0.9 | 5% | 2.22 |
| **All** | 22 | c3hen input | 0.977 / 0.978 | 3.8 / 0.7 | 2.3 | 8% | 2.21 |
| **All** | 22 | chss input | 0.964 / 0.964 | 0.3 / 0.3 | 1.0 | 4% | 2.21 |

| Phase | Model | GATC ↑ | TCD px ↓ |
|---|---|---:|---:|
| 0-6 s | c3hen | 0.488 | 77.0 |
| 0-6 s | chss | 0.429 | 58.2 |
| 6-12 s | c3hen | 0.257 | 102.1 |
| 6-12 s | chss | 0.276 | 78.9 |
| 12-18 s | c3hen | 0.212 | 123.6 |
| 12-18 s | chss | 0.217 | 100.6 |
| 18-24 s | c3hen | 0.189 | 128.7 |
| 18-24 s | chss | 0.201 | 115.6 |
| 24-31 s | c3hen | 0.149 | 150.5 |
| 24-31 s | chss | 0.164 | 124.4 |

Paired differences (c3hen-chss; win rate = share of rollouts where c3hen is better):

| Dataset | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | gatc | -0.0960 | ±0.0680 | 17% | 0.062 |
| cmr_surgical/cholecystectomy | tcd | +53.6379 | ±49.6354 | 33% | 0.16 |
| cmr_surgical/hysterectomy | gatc | -0.0875 | ±0.1534 | 50% | 0.44 |
| cmr_surgical/hysterectomy | tcd | +44.4805 | ±42.1262 | 0% | 0.12 |
| cmr_surgical/inguinal_hernia | gatc | +0.0525 | ±0.0662 | 70% | 0.28 |
| cmr_surgical/inguinal_hernia | tcd | -4.9750 | ±36.7119 | 70% | 0.38 |
| cmr_surgical/prostatectomy | gatc | -0.0069 | ±0.0276 | 50% | 0.5 |
| cmr_surgical/prostatectomy | tcd | +34.4447 | ±21.9273 | 23% | 0.0059 |
| **All** | gatc | -0.0165 | ±0.0329 | 50% | 0.4 |
| **All** | tcd | +29.4714 | ±16.8457 | 32% | 0.0031 |
