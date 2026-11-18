# accl-hbw

| Field | Value |
| --- | --- |
| Device | HBW, 80 CUs, compute capability 9.3 |
| Memory | 68.7 GB |
| Backend | compat 6.3.26093 |
| PyTorch | 2.9.0 |
| Triton | triton==3.3.0+vendor.build |
| Platform | Linux-5.15.0-25-generic-x86_64-with-glibc2.35 |
| Suite | probe,gemm,vec,bandwidth,resnet,gpt,llama,moe,decode · amp fp16 · compare_compile · full scale |
| Rows | 57 total, 57 measured, 3 errors |
| Provenance | vendor identifiers redacted, measurements untouched (see `meta.redaction`) |

## Peak (measured)

| Metric | Value | Unit |
| --- | --- | --- |
| device stream | 1,354.8 | GB/s |
| host link | 45.116 | GB/s |
| matmul bf16 | 308.3 | TFLOPS |
| matmul fp16 | 307.8 | TFLOPS |
| matmul fp32 | 55.365 | TFLOPS |
| matmul int8 | 523.8 | TOPS |
| packed vector bf16 | 1.392 | TFLOPS |
| packed vector fp16 | 26.402 | TFLOPS |
| packed vector fp32 | 25.579 | TFLOPS |
| vector bf16 | 10.385 | TFLOPS |
| vector fp16 | 7.281 | TFLOPS |
| vector fp32 | 10.753 | TFLOPS |

## probe (13)

| Config | Result | Status | Note |
| --- | --- | --- | --- |
| autocast_fp16 | 1 | pass |  |
| autocast_bf16 | 1 | pass |  |
| bf16_matmul | 1 | pass |  |
| sdpa_flash | 0 | fail | error: RuntimeError: No available kernel. Aborting execution. |
| sdpa_efficient | 0 | fail | error: RuntimeError: No available kernel. Aborting execution. |
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
| fp32 1024x1024 | 36.743 | TFLOPS | 66.4% | 0.4 |  |
| fp32 8192x8192 | 55.365 | TFLOPS | 100.0% | 0.1 |  |
| fp32 m1024 k4096 n11008 | 52.615 | TFLOPS | 95.0% | 0 |  |
| fp16 1024x1024 | 137.4 | TFLOPS | 44.6% | 0.05 |  |
| fp16 8192x8192 | 307.8 | TFLOPS | 100.0% | 0 |  |
| fp16 m1024 k4096 n11008 | 294.2 | TFLOPS | 95.6% | 0.01 |  |
| bf16 1024x1024 | 130.8 | TFLOPS | 42.4% | 0.07 |  |
| bf16 8192x8192 | 308.3 | TFLOPS | 100.0% | 0 |  |
| bf16 m1024 k4096 n11008 | 294.3 | TFLOPS | 95.4% | 0.01 |  |

## gemv (3)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 m1 k4096 n11008 | 1,167.8 | GB/s | 86.2% | 0.03 |  |
| fp16 m1 k4096 n11008 | 1,218.8 | GB/s | 90.0% | 0.22 |  |
| bf16 m1 k4096 n11008 | 1,214.7 | GB/s | 89.7% | 0.1 |  |

## gemm_int8 (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 4096x4096 | 523.8 | TOPS | 100.0% | 0.48 |  |

## gemv_int8 (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 m32 k4096 n11008 | 1,023.8 | GB/s | 75.6% | 1.31 |  |

## vec_flops (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 chain256 | 10.753 | TFLOPS | 100.0% | 1.12 |  |
| fp16 chain256 | 7.281 | TFLOPS | 100.0% | 0.41 |  |
| bf16 chain256 | 10.385 | TFLOPS | 100.0% | 1.24 |  |
| fp32 chain256 triton-native | 25.579 | TFLOPS | 100.0% | 0.05 |  |
| fp16 chain256 triton-native | 26.402 | TFLOPS | 100.0% | 0.01 |  |
| bf16 chain256 triton-native | 1.392 | TFLOPS | 100.0% | 0.03 |  |

## bandwidth (10)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| read 16MiB | 695.8 | GB/s | 51.4% | 2.23 |  |
| write 16MiB | 1,076.1 | GB/s | 79.4% | 4.37 |  |
| copy 16MiB | 1,187.3 | GB/s | 87.6% | 0.64 |  |
| h2d 16MiB | 6.274 | GB/s | 13.9% | 1.96 |  |
| d2h 16MiB | 6.099 | GB/s | 13.5% | 0.36 |  |
| read 1024MiB | 1,240.7 | GB/s | 91.6% | 0.22 |  |
| write 1024MiB | 1,223.3 | GB/s | 90.3% | 0.01 |  |
| copy 1024MiB | 1,354.8 | GB/s | 100.0% | 0.01 |  |
| h2d 1024MiB | 6.297 | GB/s | 14.0% | 10.01 |  |
| d2h 1024MiB | 45.116 | GB/s | 100.0% | 0.42 |  |

## resnet50_infer (4)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| bs1 fp16 autocast eager | 87.13 | img/s | — | 0.16 |  |
| bs1 fp16 autocast compile | 171.8 | img/s | — | 0.32 | compile ×1.972 |
| bs32 fp16 autocast eager | 2,766.8 | img/s | — | 0.3 |  |
| bs32 fp16 autocast compile | 5,266.3 | img/s | — | 0.31 | compile ×1.903 |

## gpt_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 61,545 | tok/s | 16.0% | 0.43 |  |
| L12 d768 bs8x512 fp16 autocast compile | 75,014 | tok/s | 19.5% | 0.83 | compile ×1.219 |

## gpt_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 323,813 | tok/s | 28.1% | 0.09 |  |
| L12 d768 bs8x512 fp16 autocast compile | 433,693 | tok/s | 37.6% | 0.26 | compile ×1.339 |

## llama_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L16 d2048 bs4x2048 fp16 autocast eager | 69,320 | tok/s | 47.0% | 0.06 |  |
| L16 d2048 bs4x2048 fp16 autocast compile | 92,816 | tok/s | 62.9% | 0.03 | compile ×1.339 |

## llama_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L16 d2048 bs1 fp16 | 52.394 | tok/s | 7.0% | 0.95 |  |

## moe_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| sparse E8K2 L4 d512 fp16 autocast eager | 19,094 | tok/s | — | 1.21 |  |
| sparse E8K2 L4 d512 fp16 autocast compile | 19,048 | tok/s | — | 1 | compile regression |

## gpt_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L24 d1024 bs1 fp16 | 65.84 | tok/s | 3.4% | 0.9 |  |

## Failures

| Test | Config | Kind | Message |
| --- | --- | --- | --- |
| probe | sdpa_flash | runtime | RuntimeError: No available kernel. Aborting execution. |
| probe | sdpa_efficient | runtime | RuntimeError: No available kernel. Aborting execution. |
| probe | sdpa_cudnn | runtime | RuntimeError: No available kernel. Aborting execution. |
