# LigerLayerNorm 昇腾适配验证报告（A2 / A3）

> 配套文档：《LigerLayerNorm 昇腾适配与性能优化设计文档（A2 / A3 / A5）》
> 关联任务：[triton-lang/triton-ascend#1800](https://github.com/triton-lang/triton-ascend/issues/1800)
> 测试时间：2026-10-21

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 节点 | A3 容器 `192.168.22.254:22005` |
| SoC（`rtGetSocVersion` 上报） | `Ascend910_9382` |
| Vector 核数 | 48 |
| UB | 192 KB（`ub_size_in_kbytes`） |
| liger_kernel | 0.8.2（editable 安装） |
| triton-ascend | 3.2.2 |
| CANN | 9.1.0 |
| torch_npu | 2.10.0.post4 |
| Python | 3.10.12 |

**环境干扰说明**：该节点同时运行 `sglang serve --model-path MiMo-V2-Flash-w8a8 --tp-size 8`，
16 个 die 的 HBM 均被占用约 54 GB / 64 GB，系统 load average ≈ 32。
官方 benchmark 的单次结果波动较大（同一配置曾出现 1.02 / 1.19 / 2.39），
因此本文核心结论采用**同窗口交替 A/B + 多次中位数**测量；官方 benchmark 结果见 §5。

---

## 2. 功能验证

```
$ pytest test/transformers/test_layer_norm.py -q
16 passed, 0 failed, 0 error
```

用例覆盖：`test_liger_layer_norm`（fwd/bwd 数值 + 梯度）与 `test_liger_layer_norm_functional`
（functional 接口 vs autograd Function），各 5 组与 3 组 shape，dtype 覆盖 fp32 / bf16。
**未修改任何断言与容差。**

---

## 3. 精度验证

对比 `torch.nn.functional.layer_norm`，相对误差按梯度量级归一化：

| H | dtype | y maxerr | dx maxerr | dγ/dβ rel |
|---|---|---|---|---|
| 1023 | fp32 | 5.7e-7 | 2.9e-6 | 7.6e-5 |
| 1500 | fp32 | 1.9e-6 | 2.9e-6 | 6.1e-5 |
| 2048 | bf16 | 1.6e-2 | 6.2e-2 | 4.5e-3 |
| 4096 | bf16 | 3.1e-2 | 1.6e-2 | 2.3e-3 |
| 7168 | bf16 | 3.1e-2 | 6.2e-2 | 4.4e-3 |
| 8192 | bf16 | 3.1e-2 | 6.2e-2 | 4.2e-3 |

说明：bf16 的 `dγ` 绝对误差可达 0.4~1.0，这是 torch 侧用 bf16 累加 `dγ` 产生的抵消误差；
本次实现用 fp32 累加后 cast 回，因此更接近真值，相对误差均在 5e-3 以内。

---

## 4. 与上游实现对比（前向）

B=1, T=4096, bf16，交替 A/B，15 次中位数，单位 ms。

| 模型 | H | torch | 上游 Ascend 后端 | 本次实现 | 本次 ratio |
|---|---|---|---|---|---|
| deepseek_v2_lite | 2048 | 0.0917 | 0.1246 (1.318) | 0.1504 | 1.640 |
| qwen2.5_7b | 3584 | 0.1293 | 0.1988 (1.444) | 0.1533 | 1.185 |
| llama_2_7b | 4096 | 0.0968 | 0.1368 (1.276) | 0.1759 | 1.818 |
| qwen2.5_14b | 5120 | 0.1334 | 0.3309 (2.222) | 0.1758 | 1.318 |
| **qwen2.5_72b** | **8192** | 0.2194 | 0.3011 (1.240) | **0.1661** | **0.757** |
| **deepseek_v3** | **7168** | 0.2654 | 0.3444 (1.202) | **0.1681** | **0.633** |

- 相对上游的改善：H=5120 从 2.222 → 1.318（消除三遍列 tile）；H=7168 从 1.202 → 0.633；H=8192 从 1.240 → 0.757。
- H ≤ 4096 相比上游有退化（如 H=2048：1.318 → 1.640），原因见 §6。

---

## 5. 官方 benchmark 结果

```
$ python benchmark_layer_norm.py --sweep-mode model_config --bt 65536 --overwrite
```

退出码 0，全部配置正常产出数据，无崩溃、无超时。
speed 共 21 个配置（forward / backward / full × 7 模型），本轮 **16 个 ratio > 1.0**。

> 注：`benchmark_layer_norm.py` 默认 `overwrite=False`，不加 `--overwrite` 时结果不会写入 CSV。

由于节点负载干扰，官方 benchmark 单次结果重复性差，逐配置数值请以 §4、§7 的交替 A/B 结果为准。

---

## 6. 端到端成本分解

### 6.1 前向（H=2048）

| 阶段 | 耗时 (ms) | 增量 |
|---|---|---|
| ① 直接调用 triton kernel（输出预分配） | 0.0799 | — |
| ② + Y / Mean / RSTD 分配 | 0.1102 | +0.0303 |
| ③ + `autograd.Function.apply` | 0.1422 | +0.0320 |
| ④ + `nn.Module.__call__` | 0.1485 | +0.0063 |
| — torch `F.layer_norm`（基准） | 0.0884 | — |

**kernel 本体 0.0799 ms 已快于 torch 的 0.0884 ms**，端到端差距全部来自框架层。

单项分配开销：`empty bf16[4096,2048]` 0.0066 ms、`empty fp32[4096]` 0.0050 ms、`view` 0.0031 ms。

### 6.2 反向（H=2048）

| 阶段 | 耗时 (ms) |
|---|---|
| `layer_norm_backward` 直接调用 | 0.1804 |
| 经 autograd 引擎 | 0.3158 |
| torch 反向 | 0.1286 |

### 6.3 框架固定成本

| 项 | triton-ascend | torch |
|---|---|---|
| 空 kernel（grid=1）单次 launch | **38~39 µs** | ~18 µs |
| autograd 前向开销 | +32 µs | — |
| autograd 反向开销 | +130~140 µs | — |

空 kernel 成本与 grid 无关（grid = 1 / 48 均为 39 µs）。源码位置：
`triton/backends/ascend/driver.py`，每次 launch 都执行
`device_res_manager_->BindDeviceToCurrentThread(false)` + `syncBlockLock` 分配 + 初始化 H2D 写入。

尝试过的规避手段：

| 手段 | 结果 |
|---|---|
| `TRITON_ENABLE_TASKQUEUE=false` | 更慢（68 µs），非解法 |
| 环境变量关闭 syncBlockLock | 不存在该开关（`grep TRITON_* driver.py` 仅 7 个，均不控制） |
| 双 kernel（stats + affine） | 慢 26%（launch 成本翻倍） |

---

## 7. 关键调优实验

### 7.1 带宽下限（H=8192, N=4096, bf16, 67 MB）

| 实验 | 耗时 |
|---|---|
| 纯 copy（1 读 + 1 写） | 45~49 µs |
| stats only（1 遍读） | 106 µs |
| affine only（1 读 + 1 写） | 68 µs |
| 单遍 fused（BLOCK=8192, grid=48） | 133 µs |
| torch LayerNorm | 217~227 µs |

### 7.2 pipeline-bound 规律

耗时随 BLOCK 增大而单调下降（循环次数越少越快）：

| BLOCK | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 循环次数（H=8192） | 8 | 4 | 2 | 1 |
| 耗时 | 465 µs | 305 µs | 216 µs | 207 µs |

**结论：tile 取 UB 允许的最大值，不为「多核」缩小 BLOCK。**

### 7.3 UB 硬约束

| n_cols | BLOCK | ROWS | 结果 |
|---|---|---|---|
| 8192 | 8192 | 1 | OK |
| 8192 | 8192 | 2 | ub overflow |
| 5120 / 7168 | 8192 | 1 | OK |
| 12288 | 16384 | 1 | ub overflow |
| 5120~8192（反向 row kernel） | 8192 | 1 | ub overflow |
| 5120~8192（反向 row kernel） | 4096 | 1 | OK |

### 7.4 grid 敏感性（前向，ms）

| grid | 24 | 48 | 96 | 192 | 384 | 1536 | 4096 |
|---|---|---|---|---|---|---|---|
| H=2048 | 0.1533 | 0.0833 | **0.0804** | 0.0801 | 0.0803 | 0.0862 | 0.1888 |
| H=4096 | 0.1856 | 0.1020 | 0.0996 | 0.0990 | 0.0994 | 0.1022 | 0.1815 |
| H=8192 | 0.2686 | 0.1455 | 0.1426 | **0.1426** | 0.1452 | 0.1608 | 0.1982 |

grid=24（一核两 program）严重串行；grid = n_rows 也明显变差。

### 7.5 反向方案对比（H=2048）

| 方案 | 耗时 (ms) | ratio |
|---|---|---|
| 部分和 buffer + host `sum(0)` | 0.1387 | 0.565 |
| `tl.atomic_add` 直接累加 | **0.0918** | **0.374** |

---

## 8. 调试中修掉的实现缺陷

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | masked 路径 `BLOCK < n_cols` 时无列循环 | 统计量被截断，误差达 1e+38 | 强制 `BLOCK >= n_cols`，否则抛错 |
| 2 | `grid > n_rows` 时空 program 不写部分和 | dW/dB 错误 | 归约只取前 `min(grid, n_rows)` 行 |
| 3 | workspace 复用留下脏数据 | 小 shape 用例失败 | 回退为每次新建 zeroed buffer |
| 4 | Triton 归约 kernel 非确定性错误 | 同一输入时对时错 | 改用 torch `sum(0)` |
| 5 | autotune 探针重复执行 atomic kernel | dW/dB 被累加 26 倍 | 探针使用独立临时 buffer |
| 6 | `benchmark_layer_norm.py` 需 `--overwrite` | 旧数据覆盖新结果 | 运行时加参数 |

---

## 9. 结论

| 验收项 | 结果 |
|---|---|
| 1. benchmark 全部配置正常产出，无崩溃、无超时 | ✅ |
| 2. 所有配置 ratio ≤ 1.0 | ❌ 21 个 speed 配置中 16 个 > 1.0 |
| 3. A2 / A3 / 950 平台覆盖 | A2/A3 已验证；A5 无设备 |
| 4. 精度不回归（pytest 全量） | ✅ 16 passed / 0 failed / 0 error |

未达成项（2）的根因：triton-ascend 每次 launch 固定 ~39 µs（torch ~18 µs）+ autograd 开销，
在 LayerNorm 这种 0.09~0.29 ms 的小算子上占比达 40%~70%。大 hidden_size（7168/8192）
因计算量足够仍可达标，证明 kernel 实现本身有效。

建议在**独占节点**复测以取得稳定的验收数值。
