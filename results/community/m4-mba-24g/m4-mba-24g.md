# m4-mba-24g

| Field | Value |
| --- | --- |
| Device | arm, — CUs, compute capability — |
| Memory | — GB |
| Backend | mps |
| PyTorch | 2.15.0 |
| Triton | — |
| Platform | macOS-27.2.1-arm64-arm-64bit-Mach-O |
| Suite | probe,gemm,vec,bandwidth,resnet,gpt,llama,moe,decode · amp fp16 · compare_compile · full scale |
| Rows | 57 total, 54 measured, 3 errors |

## Peak (measured)

| Metric | Value | Unit |
| --- | --- | --- |
| device stream | 102 | GB/s |
| matmul bf16 | 2.926 | TFLOPS |
| matmul fp16 | 2.949 | TFLOPS |
| matmul fp32 | 2.749 | TFLOPS |
| matmul int8 | 3.198 | TOPS |
| vector bf16 | 1.809 | TFLOPS |
| vector fp16 | 1.808 | TFLOPS |
| vector fp32 | 1.805 | TFLOPS |

## probe (13)

| Config | Result | Status | Note |
| --- | --- | --- | --- |
| autocast_fp16 | 1 | pass |  |
| autocast_bf16 | 1 | pass |  |
| bf16_matmul | 1 | pass |  |
| sdpa_flash | 1 | pass |  |
| sdpa_efficient | 1 | pass |  |
| sdpa_cudnn | 1 | pass |  |
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
| fp32 1024x1024 | 2.749 | TFLOPS | 100.0% | 1.19 |  |
| fp32 8192x8192 | 2.621 | TFLOPS | 95.3% | 3.33 |  |
| fp32 m1024 k4096 n11008 | 2.474 | TFLOPS | 90.0% | 6.34 |  |
| fp16 1024x1024 | 2.904 | TFLOPS | 98.5% | 2.22 |  |
| fp16 8192x8192 | 2.589 | TFLOPS | 87.8% | 8.61 |  |
| fp16 m1024 k4096 n11008 | 2.949 | TFLOPS | 100.0% | 3.39 |  |
| bf16 1024x1024 | 2.869 | TFLOPS | 98.1% | 2.64 |  |
| bf16 8192x8192 | 2.522 | TFLOPS | 86.2% | 6.2 |  |
| bf16 m1024 k4096 n11008 | 2.926 | TFLOPS | 100.0% | 2.71 |  |

## gemv (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 m1 k4096 n11008 | 104.6 | GB/s | 102.5% | 0.15 |  |
| fp32 m1 k4096 n32768 | 103 | GB/s | 100.9% | 0.08 |  |
| fp16 m1 k4096 n11008 | 107.5 | GB/s | 105.3% | 0.01 |  |
| fp16 m1 k4096 n65536 | 87.887 | GB/s | 86.1% | 0.62 |  |
| bf16 m1 k4096 n11008 | 104.3 | GB/s | 102.2% | 0.15 |  |
| bf16 m1 k4096 n65536 | 86.439 | GB/s | 84.7% | 0.03 |  |

## gemm_int8 (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 4096x4096 | 3.198 | TOPS | 100.0% | 0.37 |  |

## gemv_int8 (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 m32 k4096 n11008 | 23.489 | GB/s | 23.0% | 0.56 |  |
| int8 m32 k4096 n131072 | 24.171 | GB/s | 23.7% | 3.04 |  |

## vec_flops (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 chain256 | 1.805 | TFLOPS | 100.0% | 0.39 |  |
| fp16 chain256 | 1.808 | TFLOPS | 100.0% | 0.08 |  |
| bf16 chain256 | 1.809 | TFLOPS | 100.0% | 0.03 |  |
| fp32 chain256 triton-native | — | TFLOPS | — | — | error: RuntimeError: triton 不可用（无原生 Triton 后端的平台） |
| fp16 chain256 triton-native | — | TFLOPS | — | — | error: RuntimeError: triton 不可用（无原生 Triton 后端的平台） |
| bf16 chain256 triton-native | — | TFLOPS | — | — | error: RuntimeError: triton 不可用（无原生 Triton 后端的平台） |

## bandwidth (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| read 16MiB | 103 | GB/s | 101.0% | 1.98 |  |
| write 16MiB | 112.8 | GB/s | 110.5% | 0.44 |  |
| copy 16MiB | 99.003 | GB/s | 97.0% | 0.32 |  |
| read 1024MiB | 100 | GB/s | 98.0% | 0.24 |  |
| write 1024MiB | 102 | GB/s | 100.0% | 0.37 |  |
| copy 1024MiB | 98.566 | GB/s | 96.6% | 0.15 |  |

## resnet50_infer (4)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| bs1 fp16 autocast eager | 68.51 | img/s | — | 0.16 |  |
| bs1 fp16 autocast compile | 104.1 | img/s | — | 0.29 | compile ×1.52 |
| bs32 fp16 autocast eager | 155.6 | img/s | — | 0.23 |  |
| bs32 fp16 autocast compile | 285.4 | img/s | — | 0.47 | compile ×1.834 |

## gpt_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 2,422.1 | tok/s | 65.8% | 0.66 |  |
| L12 d768 bs8x512 fp16 autocast compile | 2,783 | tok/s | 75.6% | 0.72 | compile ×1.149 |

## gpt_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 10,207 | tok/s | 92.4% | 0.97 |  |
| L12 d768 bs8x512 fp16 autocast compile | 10,375 | tok/s | 94.0% | 0.91 | compile ×1.017 |

## llama_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L16 d2048 bs4x2048 fp16 autocast eager | 1,212.6 | tok/s | 85.7% | 1.07 |  |
| L16 d2048 bs4x2048 fp16 autocast compile | 1,305.5 | tok/s | 92.3% | 0.58 | compile ×1.077 |

## llama_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L16 d2048 bs1 fp16 | 51.632 | tok/s | 91.9% | 0.2 |  |

## moe_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| sparse E8K2 L4 d512 fp16 autocast eager | 9,703.9 | tok/s | — | 2.08 |  |
| sparse E8K2 L4 d512 fp16 autocast compile | 9,641.8 | tok/s | — | 5.77 | compile regression |

## gpt_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L24 d1024 bs1 fp16 | 123.8 | tok/s | 85.8% | 0.34 |  |

## Failures

| Test | Config | Kind | Message |
| --- | --- | --- | --- |
| vec_flops | fp32 chain256 triton-native | runtime | RuntimeError: triton 不可用（无原生 Triton 后端的平台） |
| vec_flops | fp16 chain256 triton-native | runtime | RuntimeError: triton 不可用（无原生 Triton 后端的平台） |
| vec_flops | bf16 chain256 triton-native | runtime | RuntimeError: triton 不可用（无原生 Triton 后端的平台） |
