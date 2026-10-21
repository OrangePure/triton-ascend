> **状态**：设计评审阶段（Draft）
> **关联任务**：[triton-lang/triton-ascend#1800](https://github.com/triton-lang/triton-ascend/issues/1800)（任务池 [#1796](https://github.com/triton-lang/triton-ascend/issues/1796)，编号 08-03「Liger LayerNorm 性能优化」）
> **作者**：社区任务承接人
> **更新日期**：2026-10-21

---

# LigerLayerNorm 昇腾适配与性能优化设计文档（A2 / A3 / A5）

## 1. 任务与验收标准

### 1.1 任务概述

在昇腾 NPU 上基于 Triton-Ascend，完成 Liger-Kernel v0.8.2 中 `LigerLayerNorm` 算子的亲和适配与性能优化，
前向、反向结果与 `torch.nn.LayerNorm` 对齐，性能不慢于 torch 原生实现，覆盖 A2 / A3 / A5 三类平台。

### 1.2 软件栈

| 组件 | 版本 |
|---|---|
| liger_kernel | 0.8.2 |
| triton-ascend | 3.2.2 |
| CANN | 9.1.0 |
| torch_npu | 与 CANN 9.1.0 配套版本 |

### 1.3 验收标准（摘自 issue 原文）

1. `benchmark/scripts/benchmark_layer_norm.py` 全部配置（forward / backward / full 各 `kernel_operation_mode`）
   正常产出数据，**无崩溃、无超时**；
2. **性能达标**：所有配置下 liger kernel 耗时均不慢于 baseline（**ratio ≤ 1.0**），提交优化前后逐配置对比数据表；
3. **平台覆盖**：在 **Ascend A2、A3、950** 平台上均可运行且满足上述性能标准；
   如因平台硬件差异存在限制，须在 PR 中说明原因；
4. **精度不回归**：`tests/test_layer_norm.py` 全量通过（0 failed / 0 error），**不得修改断言与容差**；
5. PR 附 profiling 根因分析、优化手段说明与复测命令及结果。

### 1.4 与 #1796 任务池的关系

`#1796` 是任务池索引，**不引入新的技术标准**。横向核对同批任务（#1800 LayerNorm / #1804 GroupNorm）正文，
任务标准五条、测试标准、验收标准逐字一致。任务池仅追加流程约定：

- 领取后 2 周内完成设计文档并评论「申请文档验收」；
- 整体完成后评论「申请验收」；
- 验收通过后在 `Ascend/triton-ascend-kernels` 的 `experimental` 分支提 PR。

---

## 2. 算子接口与约束

### 2.1 接口

| 参数 | 方向 | 类型 | shape | 说明 |
|---|---|---|---|---|
| `x` | 输入 | FP16 / BF16 / FP32 | `[B, T, H]`（实际支持任意 `...×H`） | 归一化输入 |
| `weight` | 输入 | 同 x | `[H]` | 可学习缩放 |
| `bias` | 输入 | 同 x | `[H]` | 可学习偏置 |
| `eps` | 属性 | FLOAT | - | 默认 1e-6 |
| `out` | 输出 | 同 x | `[B, T, H]` | 归一化结果 |

反向产出 `dx [B,T,H]`、`dweight [H]`、`dbias [H]`。

### 2.2 约束

- 支持任意 `hidden_size`，含非 2 的幂（测试覆盖 64 / 1023 / 1500 / 256 / 128，benchmark 覆盖 2048~7168）；
- 前向与反向均需与 `torch.nn.LayerNorm` 在容差内一致；
- **不允许修改测试用例断言与容差**。

---

## 3. 上游实现分析

### 3.1 Liger-Kernel v0.8.2 已自带 Ascend 后端

`src/liger_kernel/ops/backends/_ascend/ops/layer_norm.py` 已存在，并通过 `select_impl()` 在
`infer_device() == "npu"` 时自动替换默认 GPU 实现。实测确认：

```
LigerLayerNormFunction.__module__ == 'liger_kernel.ops.backends._ascend.ops.layer_norm'
```

**因此本任务不是从零移植，而是性能优化。**

### 3.2 现有 Ascend 实现结构

| kernel | 触发条件 | 策略 | 问题 |
|---|---|---|---|
| `_layer_norm_forward_fused_2d` | `BLOCK == n_cols` 且 UB 容纳多行 | 2D tile，单次 load，`E[x²]−mean²` | 仅适用于 2 的幂 H |
| `_layer_norm_forward_row` | 同上但只容 1 行 | grid-stride 逐行 | 同上 |
| `_layer_norm_forward_tiled` | 整行放不进 UB | **三遍**列 tile（均值 / 方差 / 仿射） | **每遍都从 GM 重读 X，共 3 遍** |
| `_layer_norm_backward_rows` | `BLOCK >= n_cols` | 逐行 + `tl.atomic_add` 累加 dW/dB | atomic 串行 |
| `_layer_norm_backward_tiled` | 列 tile | 两遍 + 逐 tile atomic | 多遍 + 原子 |

host 侧用 `_fused_forward_tile()` 按 UB × `memory_multiplier`(6.0/4.0/8.0) × 0.85 反推 tile。

### 3.3 基线实测（A2，Ascend910，B=1 / T=4096 / bf16，单位 ms）

| 模型 | H | torch fwd | liger fwd | fwd ratio | torch bwd | liger bwd | bwd ratio |
|---|---|---|---|---|---|---|---|
| deepseek_v2_lite | 2048 | 0.0946 | 0.1246 | **1.318** | 0.1517 | 0.3171 | **2.090** |
| deepseek_v3 | 7168 | 0.2866 | 0.3444 | **1.202** | 0.3208 | 0.5403 | **1.684** |
| llama_2_7b | 4096 | 0.1072 | 0.1368 | **1.276** | 0.2487 | 0.3295 | **1.325** |
| llama_3_8b | 4096 | 0.1078 | 0.1448 | **1.343** | 0.2517 | 0.3244 | **1.289** |
| qwen2.5_14b | 5120 | 0.1489 | 0.3309 | **2.222** | 0.3049 | 0.5103 | **1.674** |
| qwen2.5_72b | 7168 | 0.2428 | 0.3011 | **1.240** | 0.3774 | 0.4704 | **1.246** |
| qwen2.5_7b | 3584 | 0.1376 | 0.1988 | **1.444** | 0.2287 | 0.3419 | **1.495** |

**基线状态：21 个 speed 配置中 18 个 ratio > 1.0，不达标。**

### 3.4 根因

| 编号 | 现象 | 根因 |
|---|---|---|
| R1 | qwen2.5_14b（H=5120）fwd 慢 2.22× | `col_pow2 > n_cols` 即退回三遍列 tile，X 被读 3 次 |
| R2 | qwen2.5_7b（H=3584）fwd 慢 1.44× | 同上，H 非 2 的幂 |
| R3 | bwd 普遍慢 1.25~2.09× | 每 program 维护两个 `BLOCK_SIZE` fp32 累加器 + atomic；`memory_multiplier=8` 把 BLOCK 压小，落入多遍路径 |
| R4 | 小 H（2048）bwd 慢 2.09× | `_backward_grid_size` 无超配，atomic 串行化 |

---

## 4. 平台差异分析

| 平台 | 产品 / SoC | 架构 | UB | L1 | L0C | L2 | Cube:Vector |
|---|---|---|---|---|---|---|---|
| **A2** | Atlas 800T/800I A2、910B1~B4、910_93xx | arch22 | 192 KB | 512 KB | 128 KB | 192 MB | 1:2 |
| **A3** | Atlas 800T/800I A3、910_93 | arch22 代际 | 192 KB | 512 KB | 128 KB | 192 MB | 1:2 |
| **A5** | Ascend 950PR / 950DT（910_95 / 950） | arch35 | **248 KB** | 512 KB | **256 KB** | **128 MB** | 1:2 |

数据来源：`triton/backends/ascend/runtime/cv_autotune/hardware_consts.py`、`runtime/utils.py`
（910_95/950 判定为 `ub_size_in_kbytes=256`、`rf_size_in_kbytes=128`）。

**对 LayerNorm（纯 Vector 访存密集型、无 Cube 路径）的影响**：

| 差异 | A2 | A3 | A5 | 应对 |
|---|---|---|---|---|
| UB 容量 | 192 KB | 192 KB | 248 KB | 运行时探测，**下限以 A2 校准** |
| Vector 核数 | 按产品 | 按产品 | 按产品 | `num_vectorcore` 运行时探测 |
| 尾轴对齐 | VV 类需 32B | 同 | 同 | `BLOCK == n_cols` 且 32B 对齐才走无掩码快路径 |
| 寄存器文件 | 无 | 无 | 128 KB RF | A5 可选增强，非必需 |

**结论：一套算法 + host 侧按平台调参**，不做三份 kernel。

---

## 5. 设计方案

### 5.1 总体思路

```
host 侧平台探测（ub_kb / num_vectorcore / soc）
        │
        ▼
tile 规划器（按 UB 反推，A2 为下限）
        │
        ├── 前向 F1：单遍 2D tile，mask/no-mask 两种特化
        ├── 前向 F2：大 H 兜底（2 遍）
        └── 反向 B1：单遍 + 每 program 部分和（无 atomic）
```

三条主线：**X 只读一遍**、**非 2 的幂 H 不退化**、**反向去 atomic**。

### 5.2 前向 F1（主路径）

```python
s1 = tl.sum(xf, axis=1)                       # 行内 fp32 求和
s2 = tl.sum(xf * xf, axis=1)                  # 行内 fp32 平方和
mean = s1 * inv; var = s2 * inv - mean * mean
rstd = rsqrt(tl.maximum(var, 0.0) + eps)      # 防抵消产生负值
Y = (xf - mean[:, None]) * rstd[:, None] * W + B
```

- 单次 load、单次 pass，支持任意 `n_cols`；
- `max(var, 0)` 防 NaN；
- `BLOCK == n_cols` 且 `n_cols × dtype_size` 为 32B 整数倍时编译期消去列掩码，走无掩码特化。

### 5.3 反向 B1

- 每 program 处理若干行，`x_hat` / `wdy` / `c1` / `c2` / `dx` 全程 fp32；
- `dW` / `dB` 写 `[grid, n_cols]` fp32 部分和 buffer，host 侧 `sum(0)`，**移除 atomic**；
- 与上游 GPU 版语义一致（上游同样用 `(sm_count, n_cols)` 部分和 + `sum(0)`），跨平台统一。

### 5.4 平台参数化

```python
ub_kb  = runtime.utils.ub_size_in_kbytes     # A2/A3=192, A5=256/248
cores  = driver.utils.get_device_properties()["num_vectorcore"]
block  = largest_pow2(<= n_cols) such that block*rows*4B*8*mult*0.85 <= ub
```

**不写死平台字符串**，A2 / A3 / A5 自动走正确参数。

---

## 6. 实测调优结论（A2）

### 6.1 带宽与分解实验（H=8192, N=4096, bf16, 67 MB）

| 实验 | 耗时 | 说明 |
|---|---|---|
| 纯 copy（1 读 + 1 写） | **45~49 µs** | 本任务带宽下限 |
| stats only（1 遍读） | 106 µs | 读占比高 |
| affine only（1 读 + 1 写） | 68 µs | |
| 单遍 fused（BLOCK=8192, grid=48） | 133 µs | |
| 单遍 fused（BLOCK=4096） | 89 µs | |
| torch LayerNorm | 217~227 µs | 目标 |

### 6.2 关键规律：pipeline-bound

耗时随 BLOCK 增大而**单调下降**（循环次数越少越快）：

| BLOCK | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 循环次数（H=8192） | 8 | 4 | 2 | 1 |
| 耗时 | 465 µs | 305 µs | 216 µs | **207 µs** |

**结论：tile 取 UB 允许的最大值，不要为了多核而缩小 BLOCK。**

### 6.3 UB 硬约束（实测）

masked 单遍 kernel 在 192 KB UB 下：

| n_cols | BLOCK | ROWS | 结果 |
|---|---|---|---|
| 8192 | 8192 | 1 | **OK** |
| 8192 | 8192 | 2 | UB overflow |
| 5120 / 7168 | 8192 | 1 | **OK** |
| 12288 | 16384 | 1 | UB overflow |

即 **A2/A3 上 `BLOCK × ROWS ≤ 8192`（masked）**；A5 的 248 KB 可放宽，但以 A2 为下限即可保证三平台一致。

### 6.4 grid 扫描结果（前向）

| H | BLOCK | 最优 grid | liger | torch | ratio |
|---|---|---|---|---|---|
| 2048 | 2048 | 128 | 58.72 | 86.54 | **0.679** |
| 3584 | 4096 | 48 | 84.70 | 125.67 | **0.674** |
| 4096 | 4096 | 48 | 74.88 | 95.73 | **0.782** |
| 5120 | 8192 | 96 | 153.75 | 130.70 | **1.176** |
| 7168 | 8192 | 96 | 155.96 | 262.56 | **0.594** |
| 8192 | 8192 | 192 | 137.24 | 217.00 | **0.632** |

**5/6 个配置达标，H=5120 仍差 17.6%（见 7.2 待解决项）。**

---

## 7. 精度设计（不可退让项）

1. 统计量与反向归约全程 fp32；fp16/bf16 载入后立即 `.to(tl.float32)`，store 前转回；
2. `var = E[x²] − mean²` 与 0 取 `tl.maximum` 后再加 eps；
3. `rsqrt` 使用 `triton.language.math.rsqrt`（NPU 路径）；A5 若 lowering 异常退化为 `1/tl.sqrt(...)`；
4. `dW/dB` 用 fp32 部分和归约，结束后一次性 cast 回权重 dtype；
5. **不改断言、不改容差、不 skip 用例**。

实测精度（bf16，对比 `torch.nn.functional.layer_norm`）：

| H | fwd maxerr | dx maxerr | dW maxerr |
|---|---|---|---|
| 2048 | 1.56e-2 | 1.25e-1 | 1.0（torch 侧 bf16 抵消累加所致，fp32 口径 7.3e-5） |
| 4096 | 3.13e-2 | 1.56e-2 | 0.0 |
| 1023 / 1500（fp32） | 5.7e-7 | 2.9e-6 | 7.6e-5 |

---

## 8. 三平台验证矩阵

| 平台 | `pytest tests/test_layer_norm.py` | `benchmark_layer_norm.py` | 定位 |
|---|---|---|---|
| **A2**（910B3 基准卡） | 0 failed / 0 error | fwd / bwd / full × 7 模型全部 ratio ≤ 1.0 | **性能结论主口径** |
| **A3**（910_93） | 同上 | 同上 | 证明可运行且达标 |
| **A5**（950PR/950DT） | 同上 | 同上；硬件差异须在 PR 说明 | 证明可运行且达标 |

---

## 9. 环境可行性与已知问题

| 问题 | 现象 | 解决 |
|---|---|---|
| git 全局 `insteadOf` 指向失效代理 | clone 失败 | `git -c url."https://github.com/".insteadOf= clone ...` |
| `kernels` 0.14.1 与 `huggingface_hub` 类型校验冲突 | `StrictDataclassFieldValidationError: 'import_name': str \| None` | 将 `kernels/deps.py` 的 `str \| None` 改为 `Optional[str]` |
| `peft` / `kernels` 对 hub 版本互相拉扯 | 导入失败 | 保持 `huggingface_hub 0.36.2` + 上述补丁 |

### 9.1 已验证

```
pytest test/transformers/test_layer_norm.py  →  16 passed, 0 failed, 0 error
benchmark_layer_norm.py --sweep-mode model_config --bt 65536  → 正常产出
```

### 9.2 A3 / A5 设备状态

- A3 容器（22005 / 22006）驱动 `rtGetSocVersion` 上报 `Ascend910_9382`，与 910_93 同属
  triton-ascend 的同一编译路径（`c100` / `ascend_910`），UB 同为 192 KB，tile 规划口径一致；
- A5（950PR/950DT）当前无可用设备，相关章节按硬件规格设计，**待设备到位实测**。

---

## 10. 待解决项

| 编号 | 问题 | 计划 |
|---|---|---|
| T1 | H=5120 前向 ratio 1.18 | 尝试 main+tail 无掩码切分、autotune BLOCK、或在 A5 更大 UB 下复发 |
| T2 | 反向整体重构未落地 | 按 5.3 实现 B1，目标全部 ratio ≤ 1.0 |
| T3 | A5 实测 | 待设备 |
