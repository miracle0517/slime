# slime 与 Relax 的 TransferQueue 适配对比

本文档对比当前 slime 中的 TransferQueue 适配与 Relax 中的 TransferQueue 适配。

对比依据：

- slime 当前提交：`420325fa design md`
- slime TQ 主要文件：
  - `slime/utils/transfer_queue.py`
  - `slime/ray/rollout.py`
  - `slime/ray/actor_group.py`
  - `slime/backends/megatron_utils/actor.py`
  - `train.py`
  - `train_async.py`
  - `slime/utils/arguments.py`
- Relax 当前提交：`cbb1a82 refactor(scripts): separate MODEL_DIR/DATA_DIR/EXP_DIR`
- Relax TQ 主要文件：
  - `relax/core/controller.py`
  - `relax/components/rollout.py`
  - `relax/components/actor.py`
  - `relax/components/actor_fwd.py`
  - `relax/components/advantages.py`
  - `relax/distributed/ray/rollout.py`
  - `relax/backends/megatron/actor.py`
  - `relax/utils/utils.py`
  - `relax/utils/data/stream_dataloader.py`
  - `relax/utils/arguments.py`

本文只总结当前代码中真实存在的实现，不包含未来设计假设。

## 1. 总体结论

slime 的 TransferQueue 适配是一个低侵入的可选数据平面替换：默认仍走 Ray ObjectRef，只有显式启用 `--use-transfer-queue` 后，rollout 侧把完整训练 batch 写入 TQ，actor/critic 再从 TQ 读取。训练控制流、actor/critic 的调用顺序、权重更新和大部分计算逻辑仍保持 slime 原有结构。

Relax 的 TransferQueue 适配是框架级核心能力：Controller 初始化 TQ 后，Rollout、Actor、ActorFwd、Reference、Advantages 等 Ray Serve 服务都通过 TQ 交换数据。它不仅替代 rollout-to-train 数据传输，还承担跨服务依赖、消费进度、字段就绪、流式训练、全异步流水线的一部分协同语义。

因此，两者共享同一套底层 TransferQueue 数据模型，但定位明显不同：

- slime：TQ 是现有训练流程的可选传输后端。
- Relax：TQ 是服务化/全异步 RL 流水线的数据中枢。

## 2. 相同点

### 2.1 都使用 `train_{rollout_id}` 作为 partition

两边都用同一个 partition 命名协议表达训练 step：

```text
partition_id = train_{rollout_id}
```

rollout 生产的数据写入对应 partition，训练或后续计算角色根据 step 从同一个 partition 中读取。

### 2.2 都用 `task_name` 区分消费者

两边都依赖 TransferQueue 的 `task_name` 语义表达不同消费者的消费游标。

slime 中当前主要使用：

- `actor_train`
- `critic_train`

Relax 中使用范围更广，包括：

- `train`
- `actor_train`
- `train_actor`
- `actor_log_probs`
- `ref_log_probs`
- `compute_advantages_and_returns`

共同点是：同一个 partition 可以被不同 task 独立消费，避免不同角色互相抢游标。

### 2.3 都使用相同的 TQ 初始化核心配置

两边初始化 TransferQueue 时都构造类似配置：

```text
controller.sampler
controller.polling_mode
backend.SimpleStorage.total_storage_size
backend.SimpleStorage.num_data_storage_units
```

`total_storage_size` 的核心公式也一致：

```text
rollout_batch_size * n_samples_per_prompt * (max_staleness + 1)
```

这表示 TQ 至少需要容纳 `max_staleness + 1` 个 rollout batch。

### 2.4 都复用 GRPO 分组采样和 sequence length 均衡采样

两边都使用 TransferQueue 的 sampler：

- 默认：`GRPOGroupNSampler`
- 开启 `balance_data` 时：`SeqlenBalancedSampler`

两边也都把 `total_lengths` 放进数据或 metadata 中，供 sequence length balanced sampler 使用。

### 2.5 都把 slime/Relax 的训练数据转为 TensorDict

两边都有 `dict_to_tensordict()` 类似逻辑：

- 一维 list 转普通 tensor。
- 二维变长 list 转 `torch.nested.as_nested_tensor(..., layout=torch.jagged)`。
- `rollout_routed_experts` 会从三维结构 flatten 成二维结构，以适配 jagged nested tensor。
- 多模态或 metadata 类字段会走非普通 tensor 路径。

核心目的一致：减少 Python object 传输，给 TQ 一个结构化、可采样、可分片的数据载体。

### 2.6 都在 rollout 写入后补充 `total_lengths` custom meta

两边在 `async_put()` 返回 metadata 后，都会尝试把每条 sample 的 `total_lengths` 写入 custom metadata：

```text
metadata.update_custom_meta([{"total_lengths": ...}, ...])
async_set_custom_meta(metadata)
```

这使得 TQ sampler 可以在不读取完整 token 数据的情况下进行长度相关采样。

### 2.7 都需要在 Ray worker 中连接已有 TQ

两边都不是每个 worker 独立创建新的 TQ，而是：

1. 主控侧初始化 TQ config。
2. Ray actor / Ray Serve replica 进程内调用 `tq.init(config)`。
3. 再通过 `tq.get_client()` 获取 client。

slime 把这层封装在 `connect_transfer_queue(args)`。

Relax 则在各服务或 Megatron actor 中直接调用：

```python
tq.init(config.tq_config)
data_system_client = tq.get_client()
```

### 2.8 都保留模型并行 rank 间广播

两边都只让部分 rank 直接访问 TQ，然后把数据广播给同一模型并行组内的其它 rank。

共同原因是：

- 避免所有 TP/PP/CP rank 同时消费同一个 task cursor。
- 保证模型并行组内看到一致 batch。
- 降低 TQ 元数据请求和数据拉取次数。

## 3. 不同点

### 3.1 TQ 在系统中的定位不同

| 维度 | slime | Relax |
| --- | --- | --- |
| 默认行为 | 默认关闭，需要 `--use-transfer-queue` | TQ 是核心数据系统，Controller 初始化 |
| 主要目标 | 替换 rollout-to-training 的 Ray ObjectRef 数据传输 | 支撑服务化、colocate、fully async 多角色流水线 |
| 控制流 | 保留 `train.py` / `train_async.py` 原有 driver loop | Ray Serve Controller 管理多个长期运行服务 |
| 数据平面范围 | rollout -> actor/critic | rollout -> actor_fwd/ref -> advantages -> actor，也包括 critic/actor 等路径 |

slime 的实现更像一个适配层；Relax 的实现则是架构核心。

### 3.2 初始化位置不同

slime：

- `train.py` / `train_async.py` 在创建 placement group 后调用 `initialize_transfer_queue(args)`。
- RolloutManager 和 MegatronTrainRayActor 初始化时通过 `connect_transfer_queue(args)` 连接 TQ。
- `RayTrainGroup` 会把 TQ 相关 env 注入 Ray actor runtime env。

Relax：

- `Controller.__init__()` 中调用 `_initialize_data_system()`。
- Controller 生成 `config.tq_config` 后，各 Ray Serve service 和 Megatron worker 自行 `tq.init(config.tq_config)`。
- TQ 与 Ray Serve 服务生命周期绑定更紧。

### 3.3 是否可选不同

slime：

- 新增 `--use-transfer-queue`。
- 不开启时不 import 外部 `transfer_queue` 包。
- 原 Ray ObjectRef 路径完整保留。

Relax：

- 没有等价的 `--use-transfer-queue` 开关。
- TQ 是默认数据系统。
- 即使是非 fully async / colocate 模式，数据也通过 TQ 流转。

### 3.4 rollout 写入粒度不同

slime：

- `RolloutManager.generate()` 先生成完整 rollout 数据。
- `_convert_samples_to_train_data()` 转成完整 train_data。
- `transfer_rollout_data()` 一次性写入当前 step 的完整 batch。
- TQ 模式下 `generate()` 返回 `None`，driver 不再携带 Ray ObjectRef 数据。

Relax：

- rollout 生成过程中会按 batch/chunk 调用 `transfer_batch_to_data_system()`。
- 写入任务通过 `asyncio.create_task()` 后台执行，生成与写入可重叠。
- fully async 下 transfer batch size 与 `global_batch_size / num_iters_per_train_update / n_samples_per_prompt` 相关。
- colocate 下可以退化为按 rollout batch 写入。

因此 slime 是 step 级完整写入，Relax 支持 rollout 过程中的增量写入。

### 3.5 训练读取方式不同

slime：

- `MegatronTrainRayActor.train()` 中判断是否启用 TQ。
- 启用时调用 `_get_rollout_data_from_transfer_queue()`。
- `_get_rollout_data_from_transfer_queue()` 循环调用 `get_data_from_transfer_queue()`，直到拿到非空 batch。
- 每个 actor/critic step 基本按一个 DP shard 读取完整训练数据。
- 读取后仍复用原来的 `train_actor()` / `train_critic()`。

Relax：

- 普通同步路径中，Megatron actor 用 `_get_data_from_transfer_queue()` 按 `batch_index` 读取。
- fully async actor 训练使用 `create_stream_dataloader()` 创建 `StreamingDataset` / `StreamingDataLoader`。
- actor 的训练迭代可以直接从 TQ 流式拉取 microbatch。
- ActorFwd、Reference、Advantages 等服务也会从 TQ 读取输入并把输出写回 TQ。

因此 slime 读取是一次性拉取后训练，Relax 支持流式 dataloader 和多阶段读写。

### 3.6 task graph 复杂度不同

slime 当前只有 rollout 数据消费：

```text
Rollout -> TQ -> Actor/Critic
```

PPO with critic 时，critic 的 values 仍通过 Ray ObjectRef `external_data` 传给 actor。

Relax fully async 的数据依赖更像：

```text
Rollout -> TQ -> ActorFwd
Rollout -> TQ -> Reference
ActorFwd/Reference -> TQ -> Advantages
Advantages -> TQ -> Actor
Actor -> clear partition
```

Relax 不只是传原始 rollout 数据，还把 `log_probs`、`ref_log_probs`、`advantages`、`returns` 等中间结果写回同一个数据系统。

### 3.7 staleness/backpressure 语义不同

slime：

- 在写入前调用 `wait_for_staleness()`。
- 读取当前所有 `train_` partition。
- 如果 `len(train_partitions) <= max_staleness`，允许写入。
- 判断标准是未清理 partition 数量。

Relax：

- rollout step 结束后检查 partition list。
- `satisfy_staleness(partition_list, current_rollout_id, max_staleness)` 使用当前 rollout id 与最老 partition id 的距离判断。
- fully async 场景还会结合 production status 判断最终 step 或权重更新时机。

slime 的 backpressure 更简单，按数量控制；Relax 更关注 step 距离和生产/消费状态。

### 3.8 partition 清理责任不同

slime：

- actor 训练完成后，`role == "actor"` 且 `dist.get_rank() == 0` 时调用 `clear_partition()`。
- critic-only warmup 阶段 actor 不训练，因此 driver 主动调用 `rollout_manager.clear_transfer_queue_partition()`。

Relax：

- `Actor` service 的训练循环每完成一个 step 后清理 `train_{step}`。
- `Actor.train(step, clear_data=True)` 也支持由外部调用控制是否清理。
- debug rollout only 场景中，rollout 侧会清理调试 partition。
- fully async 中清理和长期服务的 step 推进绑定。

slime 的清理逻辑散在 driver 与 actor rank 0；Relax 更偏服务级 lifecycle 管理。

### 3.9 模型并行广播实现不同

slime：

- `_broadcast_payload()` 使用 `dist.broadcast_object_list()`。
- 顺序覆盖 CP、TP、PP group。
- `rollout_routed_experts` 在写入前 flatten 成 jagged tensor，但读取广播没有单独的 NCCL tensor 优化。

Relax：

- `get_data_from_transfer_queue()` 支持 `broadcast_pp`。
- colocate 模式下跨 PP 广播；fully async 模式下各 PP stage 独立读取，不做 PP 广播。
- 对 `rollout_routed_experts` 有专门优化：从 TensorDict 中取出 jagged nested tensor 的 values/offsets，用 `dist.broadcast()` 广播 tensor，避免 `broadcast_object_list()` 大对象 pickle。
- 支持 `optimize_routing_replay` 时把 routed experts 保持在 GPU 上，减少 GPU/CPU 往返。

Relax 的广播路径更复杂，也更针对大 MoE routing 数据优化。

### 3.10 字段集合和中间产物不同

slime 默认读取字段由 `default_train_data_fields(args)` 生成，核心包括：

- `tokens`
- `total_lengths`
- `response_lengths`
- `loss_masks`
- `rewards`
- `raw_reward`
- `truncated`
- `sample_indices`
- 可选：`rollout_log_probs`、`rollout_routed_experts`、`multimodal_train_inputs`、`teacher_log_probs`
- 用户额外字段：`transfer_queue_extra_data_fields`

Relax 字段集合更多，且随服务角色变化：

- rollout 基础字段：`tokens`、`total_lengths`、`response_lengths`、`loss_masks`、`rollout_log_probs`、`rewards`、`raw_reward`
- forward 输出：`log_probs`、`ref_log_probs`
- advantage 输出：`advantages`、`returns`
- OPD 扩展：`teacher_log_probs`、`teacher_topk_token_ids`、`teacher_topk_k`、`opd_reverse_kl`
- 多模态字段：`multimodal_train_inputs`
- routing replay 字段：`rollout_routed_experts`

slime 主要读取“训练输入”；Relax 还在 TQ 中保存多阶段计算结果。

### 3.11 debug 和兼容性策略不同

slime：

- `--use-transfer-queue` 不支持 `--debug-train-only` / `--load-debug-rollout-data`。
- `--use-transfer-queue` 不支持 `--use-dynamic-global-batch-size`。
- 默认路径不依赖外部 `transfer_queue` 和 `tensordict`。

Relax：

- TQ 是默认数据系统。
- Megatron actor 中保留 `debug_train_only` 路径，可通过 `get_debug_data()` 从文件加载并训练。
- fully async 下额外限制如不支持部分 advantage normalization、megatron teacher OPD、`balance_data` 等。

slime 的策略是减少组合复杂度；Relax 的策略是把更多模式纳入服务化框架，但对 fully async 组合做额外限制。

### 3.12 权重同步与服务协同不同

slime：

- 沿用原来的 `actor_model.update_weights()`。
- `train_async.py` 仍是 driver 控制的预取式异步 rollout。
- TQ 不参与权重同步决策。

Relax：

- fully async 下结合 DCS 做异步权重同步。
- Rollout service 提供 `can_do_update_weight_for_async`、`end_update_weight` 等接口协调权重更新。
- ActorFwd/Reference/Rollout 可通过 DCS 接收 actor 权重。
- TQ 的 production/consumption 状态会影响服务推进和权重更新时机。

### 3.13 故障处理和生命周期不同

slime：

- 基本沿用现有 Ray actor 生命周期。
- TQ 侧主要有 connect、put、get、clear、close。
- 没有围绕 TQ 消费状态建立复杂恢复协议。

Relax：

- Controller 有 health manager、服务重启、pending task cancel、Ray Serve 生命周期管理。
- 各服务通过 heartbeat 更新状态。
- TQ consumption status、production status、reset_consumption 被用于恢复和推进。
- 服务停止、重启、全局重启时需要处理 TQ、Ray ObjectRef stream、Serve replica 等资源。

Relax 的 TQ 适配与容错系统耦合更深。

## 4. 关键能力对照表

| 能力 | slime TQ | Relax TQ |
| --- | --- | --- |
| 默认启用 | 否，需 `--use-transfer-queue` | 是，核心数据系统 |
| 保留 Ray ObjectRef 路径 | 是 | 否，训练数据主要走 TQ |
| partition 协议 | `train_{rollout_id}` | `train_{rollout_id}` |
| sampler | GRPO / seqlen balanced | GRPO / seqlen balanced |
| TensorDict 转换 | 有 | 有 |
| rollout 写入粒度 | step 级完整 batch | 支持增量 batch/chunk |
| actor 读取 | 一次性读取 DP shard | 可一次性读取，也可 StreamingDataLoader |
| fully async | 不实现 | 核心能力 |
| ActorFwd / Reference 独立服务 | 无 | 有 |
| Advantages 独立服务 | 无 | 有 |
| 中间结果写回 TQ | 基本无 | 有，log_probs/ref_log_probs/advantages/returns 等 |
| PPO critic values 传递 | Ray ObjectRef external_data | 可通过同步组或 TQ 多阶段流转 |
| staleness 判断 | 未清理 partition 数量 | rollout step 与最老 partition 距离/生产状态 |
| partition 清理 | actor rank 0 或 driver warmup 清理 | Actor service 生命周期清理 |
| routed experts 广播优化 | flatten 后随 payload 广播 | values/offsets 单独 NCCL tensor 广播 |
| debug_train_only + TQ | 不支持 | 有 debug data 路径 |
| dynamic global batch | TQ 模式不支持 | 有其它动态 batch 能力，但 fully async 组合受限 |
| 权重同步 | 原 slime update_weights | fully async 可用 DCS |
| 服务化容错 | 基本无 TQ 专项 | HealthManager / restart / consumption status |

## 5. slime 从 Relax 继承的核心思想

slime 当前实现明显继承了 Relax 的以下设计：

1. 用 `train_{rollout_id}` 作为训练 step partition。
2. 用 `task_name` 表达消费者身份。
3. 用 TransferQueue sampler 负责 GRPO group 和 seqlen balance。
4. 用 TensorDict + jagged nested tensor 表达变长序列。
5. 把 `total_lengths` 写入 custom meta。
6. 只让模型并行源 rank 拉取数据，再广播给其它 rank。
7. actor 训练完成后清理 partition。
8. 用 `max_staleness` 控制 rollout 生产速度。

这些是两边 TQ 适配的共同底座。

## 6. slime 主动简化的部分

相比 Relax，slime 当前实现有意识地简化了以下部分：

1. 不引入 Ray Serve 多服务拓扑。
2. 不拆 ActorFwd / Reference / Advantages 独立服务。
3. 不把 log_probs、ref_log_probs、advantages、returns 等中间结果写回 TQ。
4. 不实现 StreamingDataLoader 级别的边生产边消费。
5. 不改变 actor/critic 的主要训练函数结构。
6. 不把 TQ 与 DCS 权重同步绑定。
7. 不处理复杂的 consumption status / production status 生命周期。
8. 不支持 TQ 与 debug_train_only、dynamic global batch 的复杂组合。

这些简化让 slime 的实现更小、更容易回滚，也更贴近“替换数据传输后端”的目标。

## 7. 风险与注意点

### 7.1 task_name 语义需要统一

slime 使用 `actor_train` / `critic_train`。

Relax 中同时存在 `train`、`actor_train`、`train_actor` 等名称。Relax 的历史语义更复杂，slime 当前不应直接照搬所有 task name，否则容易引入消费游标混乱。

### 7.2 staleness 语义不是完全等价

slime 当前按 partition 数量判断 backpressure，Relax 按 rollout id 与最老 partition 的距离判断。两者在 partition 连续且清理正常时表现接近，但在跳步、恢复、残留 partition 场景中语义不同。

如果未来 slime 加入故障恢复或跳过 step，需要重新审视 staleness 判断。

### 7.3 routed experts 大对象广播仍有优化空间

slime 已在写入前 flatten `rollout_routed_experts`，但读取广播仍主要依赖 object broadcast。Relax 进一步把 jagged tensor 的 values/offsets 单独用 tensor broadcast 传输，适合大 MoE 模型。

如果 slime 未来在 Qwen3 MoE / 大 routing replay 场景下发现 `data_preprocess` 时间过高，可以优先参考 Relax 的 routed experts broadcast 优化。

### 7.4 PPO critic 路径仍未完全 TQ 化

slime 当前 critic values 仍通过 Ray ObjectRef `external_data` 传给 actor。这保持实现简单，但也意味着 PPO 的完整数据图还不是纯 TQ。

Relax 的优势是能把 log_probs、values、advantages、returns 等中间结果通过 TQ 串起来，更适合跨服务部署。

### 7.5 debug / recovery 能力差距较大

Relax 的 TQ 适配在服务化、健康检查、重启恢复方面做了更多工程处理。slime 当前实现更适合正常训练主路径，TQ 异常、partition 残留、actor 失败后的恢复策略还比较轻。

## 8. 后续演进建议

如果 slime 继续向 Relax 的 TQ 能力演进，建议按以下顺序推进：

1. 先补齐更稳健的 partition 状态观测和日志，包括 put/get/clear 耗时、partition list、batch_meta size。
2. 优化 `rollout_routed_experts` 的读取广播，参考 Relax 的 values/offsets tensor broadcast。
3. 明确 staleness 的语义是否从“partition 数量”升级为“rollout id 距离”。
4. 如果需要纯 TQ PPO，再考虑把 critic values / advantages 写回 TQ。
5. 最后再评估 StreamingDataLoader、ActorFwd、Reference、Advantages 服务化拆分。

不建议一开始就完整迁移 Relax 的 fully async 拓扑，因为那会同时改变 slime 的控制流、服务生命周期、权重同步和错误恢复模型，改动面远大于当前 TQ 适配。

## 9. 总结

slime 与 Relax 的 TransferQueue 适配共享同一套基础协议：`train_{rollout_id}` partition、`task_name` 消费者、TensorDict 数据载体、GRPO/seqlen sampler、`total_lengths` custom meta 和训练后清理 partition。

最大的差异在于抽象层级：slime 把 TQ 当作可选数据传输层，目标是用较小改动替换 Ray ObjectRef；Relax 把 TQ 当作全异步训练系统的数据中枢，用它连接多个独立服务并承载中间结果流转。

从工程策略看，slime 当前实现更适合先稳定主路径；Relax 的实现则提供了未来扩展到 streaming、fully async、多服务解耦时可以逐步参考的完整样板。
