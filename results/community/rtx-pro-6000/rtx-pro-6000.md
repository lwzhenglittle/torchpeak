# rtx-pro-6000

| Field | Value |
| --- | --- |
| Device | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 188 CUs, compute capability 12.0 |
| Memory | 102 GB |
| Backend | cuda 13.2 |
| PyTorch | 2.15.0+cu132 |
| Triton | triton==3.9.0 |
| Platform | Linux-7.0.0-43-generic-x86_64-with-glibc2.39 |
| Suite | probe,gemm,vec,bandwidth,resnet,gpt,llama,moe,decode · amp fp16 · compare_compile · full scale |
| Rows | 61 total, 61 measured, 0 errors |

## Peak (measured)

| Metric | Value | Unit |
| --- | --- | --- |
| device stream | 1,637.5 | GB/s |
| host link | 18.262 | GB/s |
| matmul bf16 | 361.5 | TFLOPS |
| matmul fp16 | 347.6 | TFLOPS |
| matmul fp32 | 70.866 | TFLOPS |
| matmul int8 | 667.6 | TOPS |
| packed vector bf16 | 117.3 | TFLOPS |
| packed vector fp16 | 116.4 | TFLOPS |
| packed vector fp32 | 100.3 | TFLOPS |
| vector bf16 | 97.862 | TFLOPS |
| vector fp16 | 95.434 | TFLOPS |
| vector fp32 | 96.457 | TFLOPS |

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
| fp32 1024x1024 | 49.266 | TFLOPS | 69.5% | 1.27 |  |
| fp32 8192x8192 | 70.866 | TFLOPS | 100.0% | 0.56 |  |
| fp32 m1024 k4096 n11008 | 66.714 | TFLOPS | 94.1% | 0.95 |  |
| fp16 1024x1024 | 220.3 | TFLOPS | 63.4% | 1.94 |  |
| fp16 8192x8192 | 347.6 | TFLOPS | 100.0% | 0.94 |  |
| fp16 m1024 k4096 n11008 | 340 | TFLOPS | 97.8% | 0.45 |  |
| bf16 1024x1024 | 222.9 | TFLOPS | 61.7% | 1.77 |  |
| bf16 8192x8192 | 361.5 | TFLOPS | 100.0% | 0.79 |  |
| bf16 m1024 k4096 n11008 | 354.3 | TFLOPS | 98.0% | 0.54 |  |

## gemv (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 m1 k4096 n11008 | 1,580.8 | GB/s | 96.5% | 0.04 |  |
| fp32 m1 k4096 n32768 | 1,642.2 | GB/s | 100.3% | 0.06 |  |
| fp16 m1 k4096 n11008 | 5,038.9 | GB/s | — | 0.07 | L2-resident (cache bandwidth) |
| fp16 m1 k4096 n65536 | 1,634.7 | GB/s | 99.8% | 0.04 |  |
| bf16 m1 k4096 n11008 | 5,041.2 | GB/s | — | 0.1 | L2-resident (cache bandwidth) |
| bf16 m1 k4096 n65536 | 1,638.8 | GB/s | 100.1% | 0.04 |  |

## gemm_int8 (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 4096x4096 | 667.6 | TOPS | 100.0% | 0.64 |  |

## gemv_int8 (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| int8 m32 k4096 n11008 | 4,307.8 | GB/s | — | 0.05 | L2-resident (cache bandwidth) |
| int8 m32 k4096 n131072 | 1,577.2 | GB/s | 96.3% | 0 |  |

## vec_flops (6)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| fp32 chain256 | 96.457 | TFLOPS | 100.0% | 1.15 |  |
| fp16 chain256 | 95.434 | TFLOPS | 100.0% | 1.38 |  |
| bf16 chain256 | 97.862 | TFLOPS | 100.0% | 1.24 |  |
| fp32 chain256 triton-native | 100.3 | TFLOPS | 100.0% | 0.19 |  |
| fp16 chain256 triton-native | 116.4 | TFLOPS | 100.0% | 0.29 |  |
| bf16 chain256 triton-native | 117.3 | TFLOPS | 100.0% | 0.03 |  |

## bandwidth (10)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| read 16MiB | 2,116.5 | GB/s | 129.2% | 1.59 |  |
| write 16MiB | 3,850.8 | GB/s | 235.2% | 0.09 |  |
| copy 16MiB | 6,836.4 | GB/s | 417.5% | 0.05 |  |
| h2d 16MiB | 18.245 | GB/s | 99.9% | 0.13 |  |
| d2h 16MiB | 18.065 | GB/s | 98.9% | 0.17 |  |
| read 1024MiB | 1,634.2 | GB/s | 99.8% | 0.02 |  |
| write 1024MiB | 1,637.5 | GB/s | 100.0% | 0.02 |  |
| copy 1024MiB | 1,465.3 | GB/s | 89.5% | 0.03 |  |
| h2d 1024MiB | 10.207 | GB/s | 55.9% | 0.19 |  |
| d2h 1024MiB | 18.262 | GB/s | 100.0% | 0.78 |  |

## resnet50_infer (4)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| bs1 fp16 autocast eager | 488.2 | img/s | — | 0.27 |  |
| bs1 fp16 autocast compile | 918.6 | img/s | — | 1.22 | compile ×1.882 |
| bs32 fp16 autocast eager | 10,931 | img/s | — | 0.22 |  |
| bs32 fp16 autocast compile | 15,531 | img/s | — | 0.12 | compile ×1.421 |

## gpt_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 158,212 | tok/s | 36.5% | 1.11 |  |
| L12 d768 bs8x512 fp16 autocast compile | 162,068 | tok/s | 37.4% | 1.16 | compile ×1.024 |

## gpt_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L12 d768 bs8x512 fp16 autocast eager | 816,647 | tok/s | 62.7% | 0.38 |  |
| L12 d768 bs8x512 fp16 autocast compile | 852,377 | tok/s | 65.5% | 0.23 | compile ×1.044 |

## llama_prefill (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L16 d2048 bs4x2048 fp16 autocast eager | 92,443 | tok/s | 55.5% | 0.76 |  |
| L16 d2048 bs4x2048 fp16 autocast compile | 135,848 | tok/s | 81.5% | 0.76 | compile ×1.47 |

## llama_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L16 d2048 bs1 fp16 | 222.9 | tok/s | 24.7% | 0.31 |  |

## moe_train (2)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| sparse E8K2 L4 d512 fp16 autocast eager | 54,184 | tok/s | — | 17.47 |  |
| sparse E8K2 L4 d512 fp16 autocast compile | 45,468 | tok/s | — | 21.88 | compile regression |

## gpt_decode (1)

| Config | Value | Unit | vs meas. peak | CV % | Note |
| --- | --- | --- | --- | --- | --- |
| L24 d1024 bs1 fp16 | 387.9 | tok/s | 16.8% | 0.93 |  |
