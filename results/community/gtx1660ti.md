# lwzheng-5700x_cuda

| Field | Value |
| --- | --- |
| Device | NVIDIA GeForce GTX 1660 Ti, 24 CUs, compute capability 7.5 |
| Memory | 6 GB |
| Backend | cuda 13.2 |
| PyTorch | 2.15.0+cu132 |
| Triton | triton==3.9.0 |
| Platform | Linux-7.0.0-38-generic-x86_64-with-glibc2.43 |
| Suite | probe,gemm,vec,bandwidth,resnet,gpt,llama,moe,decode · amp fp16 · compare_compile · full scale |
| Rows | 57 total, 57 measured, 2 errors |

## Peak (measured)

| Metric | Value | Unit |
| --- | --- | --- |
| device stream | 270.3 | GB/s |
| host link | 13.123 | GB/s |
| matmul bf16 | 2.958 | TFLOPS |
| matmul fp16 | 0.752 | TFLOPS |
| matmul fp32 | 3.157 | TFLOPS |
| matmul int8 | 16.583 | TOPS |
| packed vector bf16 | 0.716 | TFLOPS |
| packed vector fp16 | 10.94 | TFLOPS |
| packed vector fp32 | 5.507 | TFLOPS |
| vector bf16 | 0.059 | TFLOPS |
| vector fp16 | 5.486 | TFLOPS |
| vector fp32 | 5.469 | TFLOPS |

## probe (13)

| Config | Result | Status | Note |
| --- | --- | --- | --- |
| autocast_fp16 | 1 | pass |  |
| autocast_bf16 | 1 | pass |  |
| bf16_matmul | 1 | pass |  |
| sdpa_flash | 0 | fail | error: RuntimeError: No available kernel. Aborting execution. |
| sdpa_efficient | 1 | pass |  |
| sdpa_cudnn | 0 | fail | error: RuntimeError: No available kernel. Aborting execution. |
| sdpa_math | 1 | pass |  |
| grad_scaler | 1 | pass |  |
| channels_last | 1 | pass |  |
| int8_mm | 1 | pass |  |
| torch_compile_fwd | 1 | pass |  |
| torch_compile_bwd | 1 | pass |  |
| host_pin_memory | 1 | pass |  |

## gemm (9)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 1024x1024 | 3.059 | TFLOPS | 96.9% | 0.57 |  |
| fp32 8192x8192 | 3.157 | TFLOPS | 100.0% | 0.28 |  |
| fp32 m1024 k4096 n11008 | 3.138 | TFLOPS | 99.4% | 0.17 |  |
| fp16 1024x1024 | 0.553 | TFLOPS | 73.5% | 0 |  |
| fp16 8192x8192 | 0.618 | TFLOPS | 82.2% | 0 |  |
| fp16 m1024 k4096 n11008 | 0.595 | TFLOPS | 79.1% | 0 |  |
| bf16 1024x1024 | 2.894 | TFLOPS | 97.8% | 0.46 |  |
| bf16 8192x8192 | 2.958 | TFLOPS | 100.0% | 0.01 |  |
| bf16 m1024 k4096 n11008 | 2.923 | TFLOPS | 98.8% | 0.02 |  |

## gemv (3)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 m1 k4096 n11008 | 266.3 | GB/s | 98.5% | 0.01 |  |
| fp16 m1 k4096 n11008 | 192.1 | GB/s | 71.1% | 0.02 |  |
| bf16 m1 k4096 n11008 | 196.9 | GB/s | 72.9% | 0.01 |  |

## gemm_int8 (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 4096x4096 | 16.583 | TOPS | 100.0% | 0.2 |  |

## gemv_int8 (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 m32 k4096 n11008 | 119.2 | GB/s | 44.1% | 0.03 |  |

## vec_flops (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 chain256 | 5.469 | TFLOPS | 100.0% | 0.02 |  |
| fp16 chain256 | 5.486 | TFLOPS | 100.0% | 0.03 |  |
| bf16 chain256 | 0.059 | TFLOPS | 100.0% | 0.01 |  |
| fp32 chain256 triton-native | 5.507 | TFLOPS | 100.0% | 0.02 |  |
| fp16 chain256 triton-native | 10.94 | TFLOPS | 100.0% | 0.02 |  |
| bf16 chain256 triton-native | 0.716 | TFLOPS | 100.0% | 0.05 |  |

## bandwidth (10)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| read 16MiB | 244.5 | GB/s | 90.4% | 0.08 |  |
| write 16MiB | 263.3 | GB/s | 97.4% | 0.05 |  |
| copy 16MiB | 247.9 | GB/s | 91.7% | 0 |  |
| h2d 16MiB | 12.433 | GB/s | 94.7% | 0.03 |  |
| d2h 16MiB | 12.691 | GB/s | 96.7% | 0.07 |  |
| read 1024MiB | 270.3 | GB/s | 100.0% | 0.01 |  |
| write 1024MiB | 270.1 | GB/s | 99.9% | 0.01 |  |
| copy 1024MiB | 246.8 | GB/s | 91.3% | 0.06 |  |
| h2d 1024MiB | 12.911 | GB/s | 98.4% | 0.01 |  |
| d2h 1024MiB | 13.123 | GB/s | 100.0% | 0.09 |  |

## resnet50_infer (4)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| bs1 fp16 autocast eager | 54.43 | img/s | — | 0.01 |  |
| bs1 fp16 autocast compile | 57.113 | img/s | — | 0 | compile ×1.049 |
| bs32 fp16 autocast eager | 87.375 | img/s | — | 0 |  |
| bs32 fp16 autocast compile | 90.366 | img/s | — | 0.01 | compile ×1.034 |

## gpt_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 759.9 | tok/s | 81.0% | 0.02 |  |
| L12 d768 bs8x512 fp16 autocast compile | 762.4 | tok/s | 81.2% | 0.01 | compile ×1.003 |

## gpt_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 2,424.2 | tok/s | 86.0% | 0 |  |
| L12 d768 bs8x512 fp16 autocast compile | 2,430.5 | tok/s | 86.3% | 0.01 | compile ×1.003 |

## llama_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L4 d512 bs2x512 fp16 autocast eager | 9,226.8 | tok/s | 99.2% | 0.01 |  |
| L4 d512 bs2x512 fp16 autocast compile | 9,297.2 | tok/s | 100.0% | 0.01 | compile ×1.008 |

## llama_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L4 d512 bs1 fp16 | 524.4 | tok/s | 14.9% | 0.06 |  |

## moe_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| sparse E8K2 L4 d512 fp16 autocast eager | 6,092.2 | tok/s | — | 0.14 |  |
| sparse E8K2 L4 d512 fp16 autocast compile | 6,091.5 | tok/s | — | 0.15 | compile regression |

## gpt_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L24 d1024 bs1 fp16 | 170.1 | tok/s | 44.6% | 0.24 |  |

## Failures

| Test | Config | Kind | Message |
| --- | --- | --- | --- |
| probe | sdpa_flash | runtime | RuntimeError: No available kernel. Aborting execution. |
| probe | sdpa_cudnn | runtime | RuntimeError: No available kernel. Aborting execution. |
