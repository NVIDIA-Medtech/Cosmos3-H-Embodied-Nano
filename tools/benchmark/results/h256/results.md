| Group | Rollouts | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |
|---|---:|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | 128 | c3hen | 0.1970 | 0.3714 | 16.98 | 14.71 |
| CMR Versius, 7.2 s | 128 | chss | 0.1967 | 0.3771 | 16.90 | 14.60 |
| dVRK, 4.8 s | 34 | c3hen | 0.0498 | 0.8153 | 26.40 | 20.90 |
| dVRK, 4.8 s | 34 | chss | 0.0631 | 0.7801 | 24.29 | 19.05 |
| **All** | 162 | c3hen | 0.1661 | 0.4646 | 18.96 | 16.01 |
| **All** | 162 | chss | 0.1687 | 0.4617 | 18.45 | 15.54 |

| Group | Phase | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |
|---|---|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | early | c3hen | 0.1426 | 0.4693 | 19.56 | 16.48 |
| CMR Versius, 7.2 s | early | chss | 0.1314 | 0.5184 | 19.94 | 16.46 |
| CMR Versius, 7.2 s | mid | c3hen | 0.1856 | 0.3831 | 17.58 | 15.14 |
| CMR Versius, 7.2 s | mid | chss | 0.1854 | 0.3851 | 17.50 | 15.02 |
| CMR Versius, 7.2 s | late | c3hen | 0.2227 | 0.3310 | 16.21 | 14.22 |
| CMR Versius, 7.2 s | late | chss | 0.2260 | 0.3247 | 16.04 | 14.08 |
| dVRK, 4.8 s | early | c3hen | 0.0345 | 0.8674 | 31.44 | 27.24 |
| dVRK, 4.8 s | early | chss | 0.0397 | 0.8603 | 30.59 | 26.32 |
| dVRK, 4.8 s | mid | c3hen | 0.0517 | 0.8093 | 26.45 | 20.73 |
| dVRK, 4.8 s | mid | chss | 0.0641 | 0.7749 | 24.39 | 18.89 |

Paired differences (c3hen-chss; win rate = share of rollouts where c3hen is better):

| Group | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | fds | +0.0003 | ±0.0049 | 52% | 0.84 |
| CMR Versius, 7.2 s | ssim | -0.0057 | ±0.0069 | 41% | 0.066 |
| CMR Versius, 7.2 s | psnr | +0.0841 | ±0.2043 | 54% | 0.38 |
| CMR Versius, 7.2 s | dynamic_psnr | +0.1066 | ±0.1454 | 55% | 0.22 |
| dVRK, 4.8 s | fds | -0.0134 | ±0.0032 | 91% | 5e-09 |
| dVRK, 4.8 s | ssim | +0.0352 | ±0.0077 | 91% | 8.1e-09 |
| dVRK, 4.8 s | psnr | +2.1047 | ±0.6071 | 94% | 2.4e-08 |
| dVRK, 4.8 s | dynamic_psnr | +1.8438 | ±0.5338 | 97% | 6.4e-09 |
| **All** | fds | -0.0026 | ±0.0040 | 60% | 0.078 |
| **All** | ssim | +0.0029 | ±0.0062 | 52% | 0.25 |
| **All** | psnr | +0.5082 | ±0.2409 | 62% | 0.00028 |
| **All** | dynamic_psnr | +0.4712 | ±0.1933 | 64% | 2.9e-05 |

Robustness check, each model scored against its own input frames:

| Group | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | fds | -0.0039 | ±0.0049 | 59% | 0.11 |
| CMR Versius, 7.2 s | ssim | +0.0108 | ±0.0072 | 61% | 0.0017 |
| CMR Versius, 7.2 s | psnr | +0.2566 | ±0.2136 | 61% | 0.019 |
| CMR Versius, 7.2 s | dynamic_psnr | +0.2446 | ±0.1570 | 63% | 0.0024 |
| dVRK, 4.8 s | fds | -0.0130 | ±0.0032 | 94% | 2.2e-09 |
| dVRK, 4.8 s | ssim | +0.0371 | ±0.0082 | 94% | 2.9e-09 |
| dVRK, 4.8 s | psnr | +2.2476 | ±0.6075 | 94% | 2e-08 |
| dVRK, 4.8 s | dynamic_psnr | +2.0207 | ±0.5499 | 97% | 6.4e-09 |
| **All** | fds | -0.0058 | ±0.0040 | 67% | 0.00095 |
| **All** | ssim | +0.0163 | ±0.0061 | 68% | 1.1e-07 |
| **All** | psnr | +0.6745 | ±0.2449 | 68% | 9.5e-07 |
| **All** | dynamic_psnr | +0.6174 | ±0.2021 | 70% | 1.9e-08 |

| Dataset | Rollouts | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |
|---|---:|---|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 18 | c3hen | 0.1591 | 0.4694 | 18.27 | 15.92 |
| cmr_surgical/cholecystectomy | 18 | chss | 0.1619 | 0.4661 | 18.15 | 15.80 |
| cmr_surgical/hysterectomy | 34 | c3hen | 0.1592 | 0.4661 | 18.48 | 15.94 |
| cmr_surgical/hysterectomy | 34 | chss | 0.1505 | 0.4890 | 18.78 | 15.96 |
| cmr_surgical/inguinal_hernia | 32 | c3hen | 0.1825 | 0.3291 | 17.53 | 14.93 |
| cmr_surgical/inguinal_hernia | 32 | chss | 0.1942 | 0.3248 | 16.83 | 14.41 |
| cmr_surgical/prostatectomy | 44 | c3hen | 0.2523 | 0.2890 | 14.90 | 13.10 |
| cmr_surgical/prostatectomy | 44 | chss | 0.2483 | 0.2923 | 14.97 | 13.20 |
| hamlyn/needle_grasp_and_handover | 2 | c3hen | 0.0428 | 0.8735 | 26.51 | 22.07 |
| hamlyn/needle_grasp_and_handover | 2 | chss | 0.0493 | 0.8255 | 27.20 | 22.22 |
| hamlyn/peg_transfer | 2 | c3hen | 0.0438 | 0.8296 | 27.05 | 23.79 |
| hamlyn/peg_transfer | 2 | chss | 0.0463 | 0.8302 | 25.96 | 22.98 |
| hamlyn/suturing_2 | 2 | c3hen | 0.0334 | 0.8591 | 29.99 | 25.13 |
| hamlyn/suturing_2 | 2 | chss | 0.0341 | 0.8540 | 29.51 | 24.36 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer | 4 | c3hen | 0.0256 | 0.9141 | 28.84 | 22.21 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer | 4 | chss | 0.0405 | 0.8772 | 25.25 | 18.82 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer | 2 | c3hen | 0.0200 | 0.9336 | 31.17 | 23.07 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer | 2 | chss | 0.0197 | 0.9353 | 30.70 | 22.30 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction | 4 | c3hen | 0.0280 | 0.9143 | 28.06 | 20.55 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction | 4 | chss | 0.0417 | 0.8812 | 25.00 | 17.95 |
| turin/mitic_lerobot_ex_vivo | 10 | c3hen | 0.0907 | 0.6533 | 21.56 | 17.04 |
| turin/mitic_lerobot_ex_vivo | 10 | chss | 0.1106 | 0.6053 | 20.36 | 16.17 |
| ucberkeley/debridement_lerobot | 4 | c3hen | 0.0247 | 0.9333 | 29.43 | 23.02 |
| ucberkeley/debridement_lerobot | 4 | chss | 0.0358 | 0.9075 | 25.91 | 20.17 |
| ucsd/surgical_learning_dataset | 4 | c3hen | 0.0480 | 0.7872 | 26.78 | 22.22 |
| ucsd/surgical_learning_dataset | 4 | chss | 0.0675 | 0.7288 | 22.73 | 18.65 |
