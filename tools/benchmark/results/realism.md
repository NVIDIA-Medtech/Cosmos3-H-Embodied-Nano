Frames at 176 rows; FVD on 16-frame clips every 8 frames. Difference = c3hen - chss (lower is better); 95% CI and the share of 200 paired bootstrap resamples (over clips) in which c3hen is better.

| Pool | Window | Metric | c3hen | chss | c3hen input | chss input | Difference | 95% CI | Share better |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CMR Versius, 7.2 s | all | FID | 44.3 | 55.4 | 9.7 | 1.6 | -11.0 | [-13.6, -7.0] | 100% |
| CMR Versius, 7.2 s | all | FVD | 274.3 | 224.5 | 15.0 | 4.5 | +49.8 | [+23.6, +112.3] | 0% |
| dVRK, 4.8 s | all | FID | 43.9 | 44.5 | 5.1 | 3.1 | -0.6 | [-6.7, +4.8] | 70% |
| dVRK, 4.8 s | all | FVD | 173.1 | 191.4 | 10.6 | 4.4 | -18.4 | [-78.1, +34.5] | 77% |

Ratios to the recorded video (1 is ideal):

| Pool | Model | Sharpness | Motion |
|---|---|---:|---:|
| CMR Versius, 7.2 s | c3hen | 0.92 | 1.02 |
| CMR Versius, 7.2 s | chss | 0.88 | 0.87 |
| dVRK, 4.8 s | c3hen | 1.01 | 1.58 |
| dVRK, 4.8 s | chss | 0.95 | 1.43 |
