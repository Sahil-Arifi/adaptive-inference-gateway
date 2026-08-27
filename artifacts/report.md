# Adaptive Inference Gateway Benchmark Report

Generated: `2026-08-27T22:10:22.532545+00:00`

## Environment

- Operating system: `Windows-10-10.0.26200-SP0`
- Machine: `AMD64`
- Processor: `AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD`
- Logical CPUs: `16`
- Python: `3.11.16`
- PyTorch: `2.13.0+cpu`
- torchvision: `0.28.0+cpu`
- ONNX: `1.22.0`
- ONNX Runtime: `1.29.0`
- CUDA available: `False`
- CUDA version: `None`
- CUDA device: `None`

## Experiment configuration

- Completed primary cases: `32`
- Requests per case: `200`
- Warmup requests per case: `25`
- Request timeout ms: `30000.0`
- Inputs: the fixed synthetic image sequence recorded by SHA-256 in `results.json`.
- Latency percentiles: calculated from individual successful HTTP request samples; warmups are excluded.
- Throughput: successful measured responses divided by measured wall-clock duration.

## ONNX parity gate

- Passed: `True`
- ONNX SHA-256: `100db70894fa9f416718901627d1d399997d0de91d462311aafef0e002441416`
- Maximum absolute logit difference: `4.291534423828125e-06`
- Mean absolute logit difference: `6.158993402052494e-07`
- Top-1 agreement: `1.0`

## Highlights

- Best measured throughput: `onnx-direct-c8` at 171.033 successful requests/s on `cpu`.
- Best measured p95 latency: `onnx-direct-c1` at 11.403 ms on `cpu`.

## Full primary results

| Case | Device | C | Batch | Wait ms | Success | Failures | Reject | Timeout | req/s | Mean ms | p50 ms | p95 ms | p99 ms | Mean queue ms | Mean backend ms | Mean realized batch | Max realized batch | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| torch-direct-c1 | cpu | 1 | 1 | 0.0 | 200 | 0 | 0 | 0 | 68.268 | 14.617 | 14.417 | 16.211 | 19.744 | 0.080 | 8.274 | 1.000 | 1 | 200 |
| torch-direct-c8 | cpu | 8 | 1 | 0.0 | 200 | 0 | 0 | 0 | 96.338 | 81.476 | 81.474 | 85.400 | 90.305 | 63.260 | 10.148 | 1.000 | 1 | 200 |
| torch-direct-c32 | cpu | 32 | 1 | 0.0 | 200 | 0 | 0 | 0 | 64.204 | 470.342 | 131.922 | 2719.474 | 3051.627 | 4.700 | 9.768 | 1.000 | 1 | 200 |
| torch-direct-c64 | cpu | 64 | 1 | 0.0 | 200 | 0 | 0 | 0 | 64.235 | 870.294 | 651.678 | 2546.708 | 2855.566 | 3.990 | 9.752 | 1.000 | 1 | 200 |
| onnx-direct-c1 | cpu | 1 | 1 | 0.0 | 200 | 0 | 0 | 0 | 99.735 | 9.997 | 9.886 | 11.403 | 13.164 | 0.000 | 3.436 | 1.000 | 1 | 200 |
| onnx-direct-c8 | cpu | 8 | 1 | 0.0 | 200 | 0 | 0 | 0 | 171.033 | 46.023 | 42.079 | 72.784 | 99.050 | 2.810 | 3.573 | 1.000 | 1 | 200 |
| onnx-direct-c32 | cpu | 32 | 1 | 0.0 | 200 | 0 | 0 | 0 | 70.197 | 434.133 | 141.862 | 1717.494 | 2701.066 | 0.785 | 3.450 | 1.000 | 1 | 200 |
| onnx-direct-c64 | cpu | 64 | 1 | 0.0 | 200 | 0 | 0 | 0 | 65.026 | 869.087 | 655.738 | 2547.699 | 2934.968 | 0.160 | 3.516 | 1.000 | 1 | 200 |
| torch-dynamic-b8-w1-c8 | cpu | 8 | 8 | 1.0 | 200 | 0 | 0 | 0 | 108.619 | 72.890 | 70.090 | 101.956 | 107.163 | 11.535 | 20.101 | 2.857 | 7 | 70 |
| torch-dynamic-b8-w1-c32 | cpu | 32 | 8 | 1.0 | 200 | 0 | 0 | 0 | 65.186 | 465.183 | 144.583 | 2124.177 | 2866.071 | 2.970 | 10.101 | 1.093 | 5 | 183 |
| torch-dynamic-b8-w1-c64 | cpu | 64 | 8 | 1.0 | 200 | 0 | 0 | 0 | 62.139 | 907.789 | 570.182 | 2824.838 | 3115.790 | 2.945 | 10.327 | 1.111 | 7 | 180 |
| torch-dynamic-b8-w3-c8 | cpu | 8 | 8 | 3.0 | 200 | 0 | 0 | 0 | 107.366 | 73.450 | 68.720 | 107.358 | 117.032 | 10.250 | 19.736 | 2.817 | 7 | 71 |
| torch-dynamic-b8-w3-c32 | cpu | 32 | 8 | 3.0 | 200 | 0 | 0 | 0 | 67.701 | 445.932 | 161.582 | 1496.907 | 2322.595 | 3.775 | 10.520 | 1.143 | 6 | 175 |
| torch-dynamic-b8-w3-c64 | cpu | 64 | 8 | 3.0 | 200 | 0 | 0 | 0 | 64.260 | 881.594 | 739.967 | 2335.149 | 2971.360 | 3.190 | 10.149 | 1.070 | 5 | 187 |
| torch-dynamic-b16-w1-c8 | cpu | 8 | 16 | 1.0 | 200 | 0 | 0 | 0 | 100.460 | 78.433 | 77.778 | 107.409 | 131.298 | 11.570 | 19.789 | 2.740 | 7 | 73 |
| torch-dynamic-b16-w1-c32 | cpu | 32 | 16 | 1.0 | 200 | 0 | 0 | 0 | 64.563 | 468.137 | 164.569 | 1675.300 | 2097.293 | 3.730 | 10.699 | 1.130 | 5 | 177 |
| torch-dynamic-b16-w1-c64 | cpu | 64 | 16 | 1.0 | 200 | 0 | 0 | 0 | 62.325 | 911.134 | 649.240 | 2690.662 | 3137.750 | 3.430 | 10.292 | 1.087 | 5 | 184 |
| torch-dynamic-b16-w3-c8 | cpu | 8 | 16 | 3.0 | 200 | 0 | 0 | 0 | 107.526 | 73.074 | 68.900 | 106.071 | 119.940 | 10.000 | 20.724 | 2.941 | 7 | 68 |
| torch-dynamic-b16-w3-c32 | cpu | 32 | 16 | 3.0 | 200 | 0 | 0 | 0 | 60.307 | 499.152 | 173.241 | 1706.611 | 2377.100 | 1.940 | 9.983 | 1.070 | 3 | 187 |
| torch-dynamic-b16-w3-c64 | cpu | 64 | 16 | 3.0 | 200 | 0 | 0 | 0 | 62.564 | 905.353 | 716.868 | 2799.846 | 3073.879 | 3.280 | 10.155 | 1.075 | 5 | 186 |
| onnx-dynamic-b8-w1-c8 | cpu | 8 | 8 | 1.0 | 200 | 0 | 0 | 0 | 167.863 | 46.736 | 43.363 | 73.945 | 89.697 | 2.115 | 4.314 | 1.250 | 5 | 160 |
| onnx-dynamic-b8-w1-c32 | cpu | 32 | 8 | 1.0 | 200 | 0 | 0 | 0 | 78.493 | 387.818 | 141.311 | 1774.831 | 2462.068 | 1.185 | 3.575 | 1.070 | 3 | 187 |
| onnx-dynamic-b8-w1-c64 | cpu | 64 | 8 | 1.0 | 200 | 0 | 0 | 0 | 64.915 | 864.542 | 591.318 | 2679.415 | 2935.770 | 0.940 | 3.570 | 1.020 | 2 | 196 |
| onnx-dynamic-b8-w3-c8 | cpu | 8 | 8 | 3.0 | 200 | 0 | 0 | 0 | 160.796 | 48.873 | 44.638 | 78.358 | 110.187 | 2.585 | 4.505 | 1.274 | 5 | 157 |
| onnx-dynamic-b8-w3-c32 | cpu | 32 | 8 | 3.0 | 200 | 0 | 0 | 0 | 76.791 | 397.541 | 140.759 | 1484.907 | 2452.158 | 1.155 | 3.714 | 1.081 | 3 | 185 |
| onnx-dynamic-b8-w3-c64 | cpu | 64 | 8 | 3.0 | 200 | 0 | 0 | 0 | 63.137 | 902.221 | 668.861 | 2667.691 | 3065.346 | 0.475 | 3.499 | 1.010 | 2 | 198 |
| onnx-dynamic-b16-w1-c8 | cpu | 8 | 16 | 1.0 | 200 | 0 | 0 | 0 | 168.866 | 46.553 | 43.350 | 73.230 | 90.938 | 2.900 | 4.556 | 1.333 | 5 | 150 |
| onnx-dynamic-b16-w1-c32 | cpu | 32 | 16 | 1.0 | 200 | 0 | 0 | 0 | 74.524 | 408.769 | 136.836 | 1352.052 | 2044.179 | 1.170 | 3.607 | 1.064 | 3 | 188 |
| onnx-dynamic-b16-w1-c64 | cpu | 64 | 16 | 1.0 | 200 | 0 | 0 | 0 | 64.822 | 869.192 | 602.743 | 2652.760 | 2964.903 | 0.785 | 3.521 | 1.020 | 2 | 196 |
| onnx-dynamic-b16-w3-c8 | cpu | 8 | 16 | 3.0 | 200 | 0 | 0 | 0 | 169.573 | 46.407 | 43.331 | 67.041 | 88.446 | 1.495 | 4.433 | 1.250 | 3 | 160 |
| onnx-dynamic-b16-w3-c32 | cpu | 32 | 16 | 3.0 | 200 | 0 | 0 | 0 | 71.302 | 427.869 | 133.367 | 1556.754 | 2711.358 | 0.780 | 3.576 | 1.053 | 3 | 190 |
| onnx-dynamic-b16-w3-c64 | cpu | 64 | 16 | 3.0 | 200 | 0 | 0 | 0 | 64.877 | 888.086 | 648.204 | 2660.993 | 2992.683 | 0.855 | 3.539 | 1.031 | 2 | 194 |

## Observed batching trade-offs

- torch on cpu at concurrency 8: best-throughput dynamic case `torch-dynamic-b8-w1-c8` changed throughput by +12.75% and p95 latency by +19.39% relative to direct.
- torch on cpu at concurrency 32: best-throughput dynamic case `torch-dynamic-b8-w3-c32` changed throughput by +5.45% and p95 latency by -44.96% relative to direct.
- torch on cpu at concurrency 64: best-throughput dynamic case `torch-dynamic-b8-w3-c64` changed throughput by +0.04% and p95 latency by -8.31% relative to direct.
- onnx on cpu at concurrency 8: best-throughput dynamic case `onnx-dynamic-b16-w3-c8` changed throughput by -0.85% and p95 latency by -7.89% relative to direct.
- onnx on cpu at concurrency 32: best-throughput dynamic case `onnx-dynamic-b8-w1-c32` changed throughput by +11.82% and p95 latency by +3.34% relative to direct.
- onnx on cpu at concurrency 64: best-throughput dynamic case `onnx-dynamic-b8-w1-c64` changed throughput by -0.17% and p95 latency by +5.17% relative to direct.

Positive changes mean an increase; a throughput increase and a p95 increase therefore describe a throughput/latency trade-off, not an unqualified improvement.

## PyTorch versus ONNX

- Direct cpu concurrency 1: PyTorch 68.268 req/s at 16.211 ms p95; ONNX 99.735 req/s at 11.403 ms p95.
- Direct cpu concurrency 8: PyTorch 96.338 req/s at 85.400 ms p95; ONNX 171.033 req/s at 72.784 ms p95.
- Direct cpu concurrency 32: PyTorch 64.204 req/s at 2719.474 ms p95; ONNX 70.197 req/s at 1717.494 ms p95.
- Direct cpu concurrency 64: PyTorch 64.235 req/s at 2546.708 ms p95; ONNX 65.026 req/s at 2547.699 ms p95.

No backend is assumed faster; the statements above are generated from this run.

## Charts

![Throughput versus p95 latency](throughput_vs_p95.png)

![Realized batch size versus throughput](batch_efficiency.png)

## Limitations

- The synthetic inputs make serving runs reproducible but do not measure ImageNet accuracy.
- These results describe only the recorded hardware, software versions, and configuration.
- Client and server share one host, so contention and loopback transport affect measurements.
- Each configuration was measured once in a fixed case order; run-to-run variance and order effects were not quantified.
- Thermal throttling, background processes, and changing cache state may influence throughput and latency.
- This educational scheduler demonstrates production concepts; it is not a replacement for NVIDIA Triton.
- Results from differently labeled CPU and CUDA runs must not be combined as if they were one environment.
