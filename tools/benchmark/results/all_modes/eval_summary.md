Policy: mean over 4 seeds / best of 4 per sample. Action MSE in normalized action space (lower is better); baseline: the family's mean recorded trajectory.

| Family | Samples | FD PSNR ↑ | ID action MSE ↓ | Policy action MSE ↓ (mean / best) | Policy PSNR ↑ (mean / best) | Baseline action MSE |
|---|---:|---:|---:|---:|---:|---:|
| cmr_clinical | 64 | 18.56 | 0.0187 | 0.1013 / 0.0564 | 16.66 / 17.78 | 0.0913 |
| dvrk_hamlyn | 64 | 28.68 | 0.0095 | 0.0520 / 0.0353 | 23.92 / 25.09 | 0.0998 |
| dvrk_jhu | 64 | 24.46 | 0.0176 | 0.0874 / 0.0544 | 21.56 / 22.38 | 0.0986 |
| dvrk_obuda | 64 | 29.91 | 0.0057 | 0.0503 / 0.0301 | 27.56 / 28.19 | 0.0989 |
| dvrk_smarts | 64 | 21.73 | 0.0112 | 0.0721 / 0.0532 | 18.95 / 19.59 | 0.0831 |
| dvrk_stanford | 64 | 30.78 | 0.0097 | 0.0486 / 0.0308 | 26.61 / 27.54 | 0.1060 |
| dvrk_tud_grasping_retraction | 64 | 25.53 | 0.0045 | 0.0246 / 0.0187 | 25.79 / 26.09 | 0.0261 |
| dvrk_ucberkeley | 64 | 31.20 | 0.0019 | 0.0127 / 0.0072 | 24.50 / 25.95 | 0.0528 |
| dvrk_ucsd | 64 | 25.27 | 0.0058 | 0.0219 / 0.0157 | 24.44 / 25.18 | 0.0393 |
| turin_mitic | 64 | 24.76 | 0.0022 | 0.0467 / 0.0294 | 20.03 / 21.02 | 0.0512 |
| **All** | 640 | 26.09 | 0.0087 | 0.0518 / 0.0331 | 23.00 / 23.88 | 0.0747 |
