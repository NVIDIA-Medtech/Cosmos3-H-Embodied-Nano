| Group | Rollouts | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |
|---|---:|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | 128 | c3hen | 0.1927 | 0.3693 | 17.17 | 14.91 |
| CMR Versius, 7.2 s | 128 | chss | 0.1911 | 0.3771 | 17.14 | 14.85 |
| dVRK, 4.8 s | 34 | c3hen | 0.0490 | 0.8180 | 26.53 | 21.08 |
| dVRK, 4.8 s | 34 | chss | 0.0614 | 0.7840 | 24.52 | 19.33 |
| **All** | 162 | c3hen | 0.1625 | 0.4635 | 19.13 | 16.21 |
| **All** | 162 | chss | 0.1639 | 0.4625 | 18.69 | 15.79 |

| Group | Phase | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |
|---|---|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | early | c3hen | 0.1375 | 0.4876 | 19.90 | 16.81 |
| CMR Versius, 7.2 s | early | chss | 0.1246 | 0.5435 | 20.42 | 16.92 |
| CMR Versius, 7.2 s | mid | c3hen | 0.1811 | 0.3833 | 17.79 | 15.35 |
| CMR Versius, 7.2 s | mid | chss | 0.1797 | 0.3861 | 17.78 | 15.30 |
| CMR Versius, 7.2 s | late | c3hen | 0.2188 | 0.3206 | 16.36 | 14.42 |
| CMR Versius, 7.2 s | late | chss | 0.2208 | 0.3157 | 16.24 | 14.30 |
| dVRK, 4.8 s | early | c3hen | 0.0335 | 0.8766 | 31.81 | 27.58 |
| dVRK, 4.8 s | early | chss | 0.0378 | 0.8705 | 31.10 | 26.80 |
| dVRK, 4.8 s | mid | c3hen | 0.0509 | 0.8111 | 26.58 | 20.98 |
| dVRK, 4.8 s | mid | chss | 0.0624 | 0.7784 | 24.63 | 19.22 |

Paired differences (c3hen-chss; win rate = share of rollouts where c3hen is better):

| Group | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | fds | +0.0016 | ±0.0049 | 51% | 0.47 |
| CMR Versius, 7.2 s | ssim | -0.0078 | ±0.0077 | 40% | 0.031 |
| CMR Versius, 7.2 s | psnr | +0.0234 | ±0.2136 | 52% | 0.83 |
| CMR Versius, 7.2 s | dynamic_psnr | +0.0642 | ±0.1564 | 51% | 0.58 |
| dVRK, 4.8 s | fds | -0.0124 | ±0.0030 | 88% | 1e-08 |
| dVRK, 4.8 s | ssim | +0.0340 | ±0.0086 | 85% | 1.6e-08 |
| dVRK, 4.8 s | psnr | +2.0071 | ±0.6637 | 88% | 1.7e-07 |
| dVRK, 4.8 s | dynamic_psnr | +1.7518 | ±0.5978 | 88% | 1.2e-07 |
| **All** | fds | -0.0013 | ±0.0040 | 59% | 0.24 |
| **All** | ssim | +0.0010 | ±0.0068 | 49% | 0.55 |
| **All** | psnr | +0.4397 | ±0.2509 | 59% | 0.0052 |
| **All** | dynamic_psnr | +0.4184 | ±0.2047 | 59% | 0.0011 |

Robustness check, each model scored against its own input frames:

| Group | Metric | Mean difference | 95% CI | Win rate | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| CMR Versius, 7.2 s | fds | +0.0004 | ±0.0049 | 52% | 0.85 |
| CMR Versius, 7.2 s | ssim | -0.0047 | ±0.0078 | 45% | 0.19 |
| CMR Versius, 7.2 s | psnr | +0.0693 | ±0.2176 | 52% | 0.5 |
| CMR Versius, 7.2 s | dynamic_psnr | +0.0742 | ±0.1611 | 54% | 0.52 |
| dVRK, 4.8 s | fds | -0.0102 | ±0.0029 | 88% | 5.2e-08 |
| dVRK, 4.8 s | ssim | +0.0244 | ±0.0085 | 85% | 1.7e-06 |
| dVRK, 4.8 s | psnr | +1.8473 | ±0.6836 | 76% | 9.1e-06 |
| dVRK, 4.8 s | dynamic_psnr | +1.6806 | ±0.6746 | 82% | 9.1e-06 |
| **All** | fds | -0.0018 | ±0.0040 | 60% | 0.14 |
| **All** | ssim | +0.0014 | ±0.0067 | 53% | 0.47 |
| **All** | psnr | +0.4425 | ±0.2492 | 57% | 0.0043 |
| **All** | dynamic_psnr | +0.4113 | ±0.2144 | 60% | 0.0035 |

| Dataset | Rollouts | Model | FDS ↓ | SSIM ↑ | PSNR ↑ | Dynamic PSNR ↑ |
|---|---:|---|---:|---:|---:|---:|
| cmr_surgical/cholecystectomy | 18 | c3hen | 0.1555 | 0.4709 | 18.45 | 16.10 |
| cmr_surgical/cholecystectomy | 18 | chss | 0.1575 | 0.4692 | 18.40 | 16.00 |
| cmr_surgical/hysterectomy | 34 | c3hen | 0.1563 | 0.4604 | 18.63 | 16.06 |
| cmr_surgical/hysterectomy | 34 | chss | 0.1469 | 0.4865 | 19.01 | 16.14 |
| cmr_surgical/inguinal_hernia | 32 | c3hen | 0.1768 | 0.3353 | 17.79 | 15.22 |
| cmr_surgical/inguinal_hernia | 32 | chss | 0.1868 | 0.3337 | 17.14 | 14.73 |
| cmr_surgical/prostatectomy | 44 | c3hen | 0.2475 | 0.2822 | 15.05 | 13.31 |
| cmr_surgical/prostatectomy | 44 | chss | 0.2421 | 0.2865 | 15.19 | 13.46 |
| hamlyn/needle_grasp_and_handover | 2 | c3hen | 0.0427 | 0.8660 | 26.53 | 23.11 |
| hamlyn/needle_grasp_and_handover | 2 | chss | 0.0490 | 0.8206 | 27.31 | 23.47 |
| hamlyn/peg_transfer | 2 | c3hen | 0.0438 | 0.8253 | 27.03 | 23.77 |
| hamlyn/peg_transfer | 2 | chss | 0.0455 | 0.8278 | 26.11 | 23.19 |
| hamlyn/suturing_2 | 2 | c3hen | 0.0334 | 0.8572 | 29.93 | 24.86 |
| hamlyn/suturing_2 | 2 | chss | 0.0334 | 0.8564 | 29.79 | 24.44 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer | 4 | c3hen | 0.0258 | 0.9131 | 28.72 | 22.04 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/needle_transfer | 4 | chss | 0.0398 | 0.8753 | 25.41 | 18.94 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer | 2 | c3hen | 0.0199 | 0.9369 | 31.18 | 23.01 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/peg_transfer | 2 | chss | 0.0186 | 0.9422 | 31.26 | 22.88 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction | 4 | c3hen | 0.0280 | 0.9105 | 28.02 | 20.40 |
| stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk/tissue_retraction | 4 | chss | 0.0412 | 0.8766 | 25.08 | 17.94 |
| turin/mitic_lerobot_ex_vivo | 10 | c3hen | 0.0909 | 0.6439 | 21.53 | 17.04 |
| turin/mitic_lerobot_ex_vivo | 10 | chss | 0.1081 | 0.6039 | 20.54 | 16.36 |
| ucberkeley/debridement_lerobot | 4 | c3hen | 0.0219 | 0.9515 | 30.26 | 24.15 |
| ucberkeley/debridement_lerobot | 4 | chss | 0.0336 | 0.9247 | 26.38 | 20.85 |
| ucsd/surgical_learning_dataset | 4 | c3hen | 0.0436 | 0.8254 | 27.36 | 22.62 |
| ucsd/surgical_learning_dataset | 4 | chss | 0.0638 | 0.7537 | 22.98 | 18.65 |
