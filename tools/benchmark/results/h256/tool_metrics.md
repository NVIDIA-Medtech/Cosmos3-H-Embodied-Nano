| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | Missed tools | Tools per frame |
|---|---:|---|---:|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 18 | reference | | | | | 2.09 |
| cmr_surgical/cholecystectomy | 18 | c3hen | 0.413 / 0.454 | 122.3 / 96.4 | 31.6 | 34% | 1.81 |
| cmr_surgical/cholecystectomy | 18 | chss | 0.358 / 0.349 | 91.4 / 84.9 | 37.4 | 24% | 2.23 |
| cmr_surgical/hysterectomy | 34 | reference | | | | | 2.53 |
| cmr_surgical/hysterectomy | 34 | c3hen | 0.511 / 0.451 | 130.4 / 116.0 | 50.3 | 38% | 2.01 |
| cmr_surgical/hysterectomy | 34 | chss | 0.498 / 0.454 | 110.2 / 89.8 | 43.0 | 31% | 2.25 |
| cmr_surgical/inguinal_hernia | 32 | reference | | | | | 1.74 |
| cmr_surgical/inguinal_hernia | 32 | c3hen | 0.431 / 0.446 | 72.1 / 49.6 | 32.3 | 22% | 1.71 |
| cmr_surgical/inguinal_hernia | 32 | chss | 0.355 / 0.353 | 107.4 / 64.2 | 33.8 | 30% | 1.44 |
| cmr_surgical/prostatectomy | 44 | reference | | | | | 2.32 |
| cmr_surgical/prostatectomy | 44 | c3hen | 0.345 / 0.356 | 111.9 / 98.3 | 55.2 | 30% | 1.93 |
| cmr_surgical/prostatectomy | 44 | chss | 0.331 / 0.349 | 104.1 / 92.0 | 50.9 | 27% | 2.02 |
| **All** | 128 | reference | | | | | 2.20 |
| **All** | 128 | c3hen | 0.420 / 0.400 | 108.3 / 97.7 | 44.7 | 31% | 1.88 |
| **All** | 128 | chss | 0.385 / 0.363 | 104.7 / 88.9 | 42.6 | 28% | 1.97 |

Resolution control: each model's own input frames scored as if they were its prediction (one per clip):

| Dataset | Rollouts | Model | GATC ↑ (mean / median) | TCD px ↓ (mean / median) | Matched distance px | Missed tools | Tools per frame |
|---|---:|---|---:|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 9 | c3hen input | 0.858 / 0.859 | 1.8 / 1.3 | 3.7 | 8% | 2.19 |
| cmr_surgical/cholecystectomy | 9 | chss input | 0.924 / 0.924 | 0.7 / 0.6 | 1.0 | 8% | 2.06 |
| cmr_surgical/hysterectomy | 17 | c3hen input | 0.882 / 0.880 | 20.7 / 3.2 | 5.8 | 15% | 2.48 |
| cmr_surgical/hysterectomy | 17 | chss input | 0.933 / 0.933 | 16.5 / 0.6 | 2.2 | 10% | 2.50 |
| cmr_surgical/inguinal_hernia | 16 | c3hen input | 0.845 / 0.860 | 0.8 / 0.7 | 1.5 | 6% | 1.71 |
| cmr_surgical/inguinal_hernia | 16 | chss input | 0.919 / 0.921 | 0.4 / 0.4 | 0.8 | 4% | 1.72 |
| cmr_surgical/prostatectomy | 22 | c3hen input | 0.852 / 0.851 | 2.2 / 1.5 | 3.3 | 8% | 2.27 |
| cmr_surgical/prostatectomy | 22 | chss input | 0.921 / 0.919 | 0.6 / 0.6 | 1.3 | 4% | 2.34 |
| **All** | 64 | c3hen input | 0.859 / 0.861 | 6.7 / 1.3 | 3.5 | 9% | 2.17 |
| **All** | 64 | chss input | 0.924 / 0.924 | 4.8 / 0.5 | 1.4 | 6% | 2.19 |

| Phase | Model | GATC ↑ | TCD px ↓ |
|---|---|---:|---:|
| early | c3hen | 0.568 | 83.8 |
| early | chss | 0.588 | 75.0 |
| mid | c3hen | 0.457 | 102.8 |
| mid | chss | 0.426 | 105.9 |
| late | c3hen | 0.343 | 122.5 |
| late | chss | 0.316 | 118.0 |

Paired differences (c3hen-chss; win rate = share of rollouts where c3hen is better):

| Dataset | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | gatc | +0.0556 | ±0.0528 | 67% | 0.034 |
| cmr_surgical/cholecystectomy | tcd | +30.9537 | ±40.0898 | 39% | 0.15 |
| cmr_surgical/hysterectomy | gatc | +0.0130 | ±0.0358 | 56% | 0.37 |
| cmr_surgical/hysterectomy | tcd | +20.2567 | ±16.7436 | 24% | 0.011 |
| cmr_surgical/inguinal_hernia | gatc | +0.0757 | ±0.0462 | 69% | 0.0032 |
| cmr_surgical/inguinal_hernia | tcd | -35.3282 | ±27.1918 | 53% | 0.15 |
| cmr_surgical/prostatectomy | gatc | +0.0143 | ±0.0330 | 50% | 0.65 |
| cmr_surgical/prostatectomy | tcd | +7.8310 | ±12.3387 | 39% | 0.15 |
| **All** | gatc | +0.0351 | ±0.0205 | 59% | 0.0027 |
| **All** | tcd | +3.5934 | ±11.3799 | 38% | 0.064 |
