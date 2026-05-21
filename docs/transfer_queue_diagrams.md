# slime TransferQueue 类图与时序图

本文档按最新 TransferQueue 提交生成：

- commit: `d2b2b3711b821cbc3c21628bda40be30813f2363`
- subject: `transfer_queue`
- 涉及文件：
  - `train.py`
  - `train_async.py`
  - `slime/ray/placement_group.py`
  - `slime/ray/rollout.py`
  - `slime/ray/actor_group.py`
  - `slime/backends/megatron_utils/actor.py`
  - `slime/utils/arguments.py`
  - `slime/utils/transfer_queue.py`
  - `tests/utils/test_transfer_queue_utils.py`

图中只表达该 commit 中已经存在的类、函数和调用关系；外部 TransferQueue/TensorDict 只列出本 commit 实际调用到的接口名。

## 1. 实际模块类图

```mermaid
classDiagram
    direction LR

    class TrainScript {
        <<train.py>>
        +train(args)
    }

    class TrainAsyncScript {
        <<train_async.py>>
        +train(args)
    }

    class PlacementGroupModule {
        <<slime.ray.placement_group>>
        +create_placement_groups(args)
        +create_rollout_manager(args, pg)
        +create_training_models(args, pgs, rollout_manager)
        +allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role)
    }

    class RayTrainGroup {
        <<slime.ray.actor_group>>
        +__init__(args, num_nodes, num_gpus_per_node, pg, num_gpus_per_actor, role)
        +_allocate_gpus_for_actor(pg, num_gpus_per_actor)
        +async_init(args, role, with_ref, with_opd_teacher)
        +async_train(rollout_id, rollout_data_ref, external_data)
        +set_rollout_manager(rollout_manager)
    }

    class RolloutManager {
        <<Ray actor>>
        +__init__(args, pg)
        +generate(rollout_id)
        +clear_transfer_queue_partition(rollout_id)
        +dispose()
        -_get_rollout_data(rollout_id)
        -_convert_samples_to_train_data(samples)
        -_split_train_data_by_dp(data, dp_size)
    }

    class TrainRayActor {
        <<base class>>
        +__init__(world_size, rank, master_addr, master_port)
        +init(args, role, with_ref, with_opd_teacher)
        +train(rollout_id, rollout_data_ref, external_data)
        +set_rollout_manager(rollout_manager)
    }

    class MegatronTrainRayActor {
        <<Ray actor implementation>>
        +init(args, role, with_ref, with_opd_teacher)
        +train(rollout_id, rollout_data_ref, external_data)
        +train_critic(rollout_id, rollout_data)
        +train_actor(rollout_id, rollout_data, external_data)
        -_get_rollout_data(rollout_data_ref)
        -_get_rollout_data_from_transfer_queue(rollout_id)
        -_postprocess_transfer_queue_rollout_data(rollout_data)
    }

    class TransferQueueUtils {
        <<slime.utils.transfer_queue>>
        +transfer_queue_enabled(args)
        +initialize_transfer_queue(args)
        +transfer_queue_env_vars(args)
        +connect_transfer_queue(args)
        +close_transfer_queue(args)
        +add_total_lengths(train_data)
        +dict_to_tensordict(data, batch_size, device)
        +transfer_rollout_data(args, client, rollout_id, train_data)
        +wait_for_staleness(args, client)
        +clear_partition(args, client, rollout_id)
        +partition_id(rollout_id)
        +default_train_data_fields(args)
        +get_data_from_transfer_queue(args, client, rollout_id, task_name, data_fields)
        +tensordict_to_rollout_data(data)
        -_build_sampler(args, tq)
        -_set_total_length_custom_meta(client, metadata, total_lengths)
        -_broadcast_payload(payload, device)
    }

    class ArgumentParser {
        <<slime.utils.arguments>>
        +parse_args(add_custom_arguments)
        +add_transfer_queue_arguments(parser)
    }

    class ExternalTransferQueue {
        <<external package>>
        +init(conf)
        +get_client()
        +close()
        +GRPOGroupNSampler(n_samples_per_prompt)
        +SeqlenBalancedSampler(n_samples_per_prompt, dp_size)
    }

    class TransferQueueClient {
        <<external client>>
        +async_put(data, partition_id)
        +async_set_custom_meta(metadata)
        +async_get_partition_list()
        +async_clear_partition(partition_id)
        +get_meta(data_fields, batch_size, partition_id, sampling_config, task_name)
        +get_data(batch_meta)
    }

    class TensorDict {
        <<external tensordict>>
    }

    TrainScript --> TransferQueueUtils : initialize_transfer_queue
    TrainAsyncScript --> TransferQueueUtils : initialize_transfer_queue
    TrainScript --> PlacementGroupModule : create pgs / actors
    TrainAsyncScript --> PlacementGroupModule : create pgs / actors
    PlacementGroupModule --> RolloutManager : create_rollout_manager
    PlacementGroupModule --> RayTrainGroup : create_training_models
    TrainScript --> RolloutManager : generate / clear partition
    TrainAsyncScript --> RolloutManager : generate / clear partition
    TrainScript --> RayTrainGroup : async_train
    TrainAsyncScript --> RayTrainGroup : async_train

    RayTrainGroup --> TransferQueueUtils : transfer_queue_env_vars
    RayTrainGroup --> MegatronTrainRayActor : creates Ray actors
    MegatronTrainRayActor --|> TrainRayActor

    RolloutManager --> TransferQueueUtils : connect / put / clear / close
    MegatronTrainRayActor --> TransferQueueUtils : connect / get / clear
    ArgumentParser --> TransferQueueUtils : args consumed by utils

    TransferQueueUtils --> ExternalTransferQueue : import and init
    TransferQueueUtils --> TransferQueueClient : client operations
    TransferQueueUtils --> TensorDict : serialize / deserialize
```

## 2. 初始化时序图

```mermaid
sequenceDiagram
    autonumber
    participant D as train.py / train_async.py
    participant PG as create_placement_groups
    participant TQU as slime.utils.transfer_queue
    participant TQ as external transfer_queue
    participant RM as RolloutManager
    participant RTG as RayTrainGroup
    participant MTA as MegatronTrainRayActor

    D->>PG: create_placement_groups(args)
    PG-->>D: pgs
    D->>TQU: initialize_transfer_queue(args)

    alt args.use_transfer_queue is false
        TQU->>TQU: args.tq_config = None
        TQU-->>D: return
    else args.use_transfer_queue is true
        TQU->>TQU: transfer_queue_env_vars(args)
        TQU->>TQU: _build_sampler(args, tq)
        TQU->>TQ: init(conf=tq_config)
        TQ-->>TQU: tq_config or initialized config
        TQU->>TQU: args.tq_config = result
        TQU-->>D: return
    end

    D->>RM: create_rollout_manager(args, pgs["rollout"])
    RM->>TQU: connect_transfer_queue(args)
    alt TQ enabled
        TQU->>TQ: init(args.tq_config)
        TQU->>TQ: get_client()
        TQ-->>RM: transfer_queue_client
    else TQ disabled
        TQU-->>RM: None
    end

    D->>RTG: create_training_models(args, pgs, rollout_manager)
    RTG->>TQU: transfer_queue_env_vars(args)
    RTG->>MTA: create Ray actor with runtime_env env_vars
    RTG->>MTA: async_init(args, role)
    MTA->>MTA: TrainRayActor.init + Megatron init
    MTA->>TQU: connect_transfer_queue(args)
    TQU-->>MTA: transfer_queue_client or None
```

## 3. rollout 写入时序图

```mermaid
sequenceDiagram
    autonumber
    participant D as Driver
    participant RM as RolloutManager
    participant TQU as slime.utils.transfer_queue
    participant C as TransferQueueClient
    participant TQ as TransferQueue storage

    D->>RM: generate(rollout_id)
    RM->>RM: _get_rollout_data(rollout_id)
    RM->>RM: _save_debug_rollout_data(data)
    RM->>RM: _log_rollout_data(...)

    alt args.debug_rollout_only
        RM-->>D: return None
    else normal training
        RM->>RM: _convert_samples_to_train_data(samples)

        alt args.use_transfer_queue is true
            RM->>TQU: transfer_rollout_data(args, client, rollout_id, train_data)
            TQU->>TQU: wait_for_staleness(args, client)
            TQU->>TQU: add_total_lengths(train_data)
            TQU->>TQU: dict_to_tensordict(train_data, batch_size=len(tokens))
            TQU->>C: async_put(data=TensorDict, partition_id=train_{rollout_id})
            C->>TQ: write partition
            TQ-->>C: metadata
            C-->>TQU: metadata
            TQU->>TQU: _set_total_length_custom_meta(metadata, total_lengths)
            TQU->>C: async_set_custom_meta(metadata)
            TQU-->>RM: return
            RM-->>D: rollout_data_ref = None
        else args.use_transfer_queue is false
            RM->>RM: _split_train_data_by_dp(train_data, dp_size)
            RM->>RM: ray.put per DP partition
            RM-->>D: rollout_data_ref list[Box(ObjectRef)]
        end
    end
```

## 4. actor 无 critic 训练时序图

```mermaid
sequenceDiagram
    autonumber
    participant D as Driver
    participant AG as Actor RayTrainGroup
    participant A as MegatronTrainRayActor(role=actor)
    participant TQU as slime.utils.transfer_queue
    participant C as TransferQueueClient

    D->>AG: async_train(rollout_id, rollout_data_ref)
    AG->>A: train.remote(rollout_id, rollout_data_ref, external_data=None)

    opt args.offload_train
        A->>A: wake_up()
    end

    alt args.use_transfer_queue is true
        A->>A: _get_rollout_data_from_transfer_queue(rollout_id)
        loop until rollout_data is not None
            A->>TQU: get_data_from_transfer_queue(args, client, rollout_id, task_name="actor_train")
            TQU->>C: get_meta(data_fields, batch_size, partition_id, sampling_config, task_name)
            C-->>TQU: batch_meta
            alt batch_meta.size != 0
                TQU->>C: get_data(batch_meta)
                C-->>TQU: TensorDict
                TQU->>TQU: _broadcast_payload([TensorDict, batch_meta], device)
                TQU->>TQU: tensordict_to_rollout_data(TensorDict)
                TQU-->>A: RolloutBatch
            else empty meta
                TQU->>TQU: _broadcast_payload([None, None], device)
                TQU-->>A: None
            end
        end
        A->>A: _postprocess_transfer_queue_rollout_data(rollout_data)
    else args.use_transfer_queue is false
        A->>A: _get_rollout_data(rollout_data_ref)
        A->>A: process_rollout_data(...)
    end

    A->>A: train_actor(rollout_id, rollout_data)

    alt args.use_transfer_queue and role is actor and dist.get_rank()==0
        A->>TQU: clear_partition(args, client, rollout_id)
        TQU->>C: async_clear_partition(partition_id=train_{rollout_id})
    end

    opt args.offload_train
        A->>A: sleep()
    end

    A-->>AG: None
    AG-->>D: Ray refs resolved
```

## 5. PPO critic + actor 时序图

```mermaid
sequenceDiagram
    autonumber
    participant D as Driver
    participant RM as RolloutManager
    participant CG as Critic RayTrainGroup
    participant CActor as MegatronTrainRayActor(role=critic)
    participant AG as Actor RayTrainGroup
    participant A as MegatronTrainRayActor(role=actor)
    participant TQU as slime.utils.transfer_queue
    participant TQC as TransferQueueClient

    D->>CG: async_train(rollout_id, rollout_data_ref)
    CG->>CActor: train.remote(rollout_id, rollout_data_ref, external_data=None)

    alt args.use_transfer_queue is true
        CActor->>TQU: get_data_from_transfer_queue(..., task_name="critic_train")
        TQU->>TQC: get_meta(..., partition_id=train_{rollout_id}, task_name="critic_train")
        TQC-->>TQU: batch_meta
        TQU->>TQC: get_data(batch_meta)
        TQC-->>TQU: TensorDict
        TQU->>TQU: _broadcast_payload(...)
        TQU->>TQU: tensordict_to_rollout_data(...)
        TQU-->>CActor: RolloutBatch
        CActor->>CActor: _postprocess_transfer_queue_rollout_data(...)
    else args.use_transfer_queue is false
        CActor->>CActor: _get_rollout_data(rollout_data_ref)
    end

    CActor->>CActor: train_critic(rollout_id, rollout_data)
    CActor->>CActor: get_values + compute_advantages_and_returns + value_loss train
    CActor-->>CG: {"values": cpu tensors} or {}
    CG-->>D: value_refs

    alt rollout_id >= args.num_critic_only_steps
        D->>AG: async_train(rollout_id, rollout_data_ref, external_data=value_refs)
        AG->>A: train.remote(rollout_id, rollout_data_ref, external_data)
        alt args.use_transfer_queue is true
            A->>TQU: get_data_from_transfer_queue(..., task_name="actor_train")
            TQU->>TQC: get_meta(..., partition_id=train_{rollout_id}, task_name="actor_train")
            TQC-->>TQU: batch_meta
            TQU->>TQC: get_data(batch_meta)
            TQC-->>TQU: TensorDict
            TQU->>TQU: _broadcast_payload(...)
            TQU->>TQU: tensordict_to_rollout_data(...)
            TQU-->>A: RolloutBatch
            A->>A: _postprocess_transfer_queue_rollout_data(...)
        else args.use_transfer_queue is false
            A->>A: _get_rollout_data(rollout_data_ref)
        end
        A->>A: train_actor(rollout_id, rollout_data, external_data=value_refs)
        alt args.use_transfer_queue and dist.get_rank()==0
            A->>TQU: clear_partition(args, client, rollout_id)
            TQU->>TQC: async_clear_partition(train_{rollout_id})
        end
        A-->>AG: None
        AG-->>D: actor train done
    else critic-only warmup
        D->>D: ray.get(value_refs)
        alt args.use_transfer_queue is true
            D->>RM: rollout_manager.clear_transfer_queue_partition(rollout_id)
        end
    end
```

## 6. critic-only warmup partition 清理时序图

```mermaid
sequenceDiagram
    autonumber
    participant D as train.py / train_async.py
    participant CG as Critic RayTrainGroup
    participant CActor as MegatronTrainRayActor(role=critic)
    participant RM as RolloutManager
    participant TQU as slime.utils.transfer_queue
    participant TQC as TransferQueueClient

    D->>CG: async_train(rollout_id, rollout_data_ref)
    CG->>CActor: train.remote(rollout_id, rollout_data_ref)
    CActor->>CActor: train_critic(...)
    CActor-->>CG: value_refs result
    CG-->>D: value_refs resolved

    Note over D,TQC: actor 不训练，因此 MegatronTrainRayActor(role=actor) 不会触发 clear_partition

    alt args.use_transfer_queue is true
        D->>RM: clear_transfer_queue_partition.remote(rollout_id)
        RM->>TQU: clear_partition(args, transfer_queue_client, rollout_id)
        TQU->>TQC: async_clear_partition(partition_id=train_{rollout_id})
        TQC-->>TQU: clear done
        TQU-->>RM: return
        RM-->>D: clear done
    else args.use_transfer_queue is false
        D->>D: no partition cleanup
    end
```

## 7. TransferQueue 读取与模型并行广播时序图

```mermaid
sequenceDiagram
    autonumber
    participant Source as TP0 PP0 CP0 rank
    participant Other as Other MP ranks
    participant TQU as slime.utils.transfer_queue
    participant TQC as TransferQueueClient
    participant MP as Megatron parallel groups

    Source->>TQU: get_data_from_transfer_queue(args, client, rollout_id, task_name)
    Other->>TQU: get_data_from_transfer_queue(args, client, rollout_id, task_name)

    TQU->>TQU: data_fields = default_train_data_fields(args)
    TQU->>TQU: batch_size = rollout_batch_size * n_samples_per_prompt / DP size
    TQU->>TQU: sampling_config = {dp_rank, task_name, batch_index=0, partition_id}

    alt rank is TP0 and PP0 and CP0
        TQU->>TQC: get_meta(data_fields, batch_size, partition_id, sampling_config, task_name)
        TQC-->>TQU: batch_meta
        alt batch_meta.size != 0
            TQU->>TQC: get_data(batch_meta)
            TQC-->>TQU: TensorDict
            TQU->>TQU: payload = [TensorDict, batch_meta]
        else empty batch
            TQU->>TQU: payload = [None, None]
        end
    else non-source model-parallel rank
        TQU->>TQU: payload = [None, None]
    end

    opt context_parallel_world_size > 1
        TQU->>MP: broadcast_object_list(payload, context_parallel_group)
        MP-->>Other: payload
    end

    TQU->>MP: broadcast_object_list(payload, tensor_model_parallel_group)
    MP-->>Other: payload

    opt pipeline_model_parallel_world_size > 1
        TQU->>MP: broadcast_object_list(payload, pipeline_model_parallel_group)
        MP-->>Other: payload
    end

    alt payload has TensorDict
        TQU->>TQU: tensordict_to_rollout_data(TensorDict)
        TQU-->>Source: RolloutBatch, batch_meta
        TQU-->>Other: RolloutBatch, batch_meta
    else payload empty
        TQU-->>Source: None, None
        TQU-->>Other: None, None
    end
```

## 8. staleness backpressure 时序图

```mermaid
sequenceDiagram
    autonumber
    participant RM as RolloutManager
    participant TQU as slime.utils.transfer_queue
    participant TQC as TransferQueueClient

    RM->>TQU: transfer_rollout_data(args, client, rollout_id, train_data)
    TQU->>TQU: wait_for_staleness(args, client)

    loop until train partition count <= args.max_staleness
        TQU->>TQC: async_get_partition_list()
        TQC-->>TQU: partitions
        TQU->>TQU: filter partition startswith "train_"
        alt len(train_partitions) > args.max_staleness
            TQU->>TQU: sleep(args.transfer_queue_staleness_poll_interval)
        else writable
            TQU->>TQU: return from wait_for_staleness
        end
    end

    TQU->>TQU: add_total_lengths(train_data)
    TQU->>TQU: dict_to_tensordict(train_data, batch_size=len(tokens))
    TQU->>TQC: async_put(data=TensorDict, partition_id=train_{rollout_id})
    TQC-->>TQU: metadata
    TQU->>TQC: async_set_custom_meta(metadata with total_lengths)
    TQU-->>RM: return
```

## 9. 参数校验与字段选择图

```mermaid
flowchart TD
    Args[parse_args final validation] --> TQArg{use_transfer_queue}
    TQArg -- false --> TQConfig[tq_config = None if missing]
    TQArg -- true --> CheckDebug{debug_train_only}
    CheckDebug -- true --> ErrorDebug[raise ValueError]
    CheckDebug -- false --> CheckDGBS{use_dynamic_global_batch_size}
    CheckDGBS -- true --> ErrorDGBS[raise ValueError]
    CheckDGBS -- false --> CheckStale{max_staleness >= 0}
    CheckStale -- false --> ErrorStale[raise ValueError]
    CheckStale -- true --> CheckStorage{num_data_storage_units > 0}
    CheckStorage -- false --> ErrorStorage[raise ValueError]
    CheckStorage -- true --> Valid[TransferQueue args valid]

    Fields[default_train_data_fields] --> Base[tokens total_lengths response_lengths loss_masks rewards raw_reward truncated sample_indices]
    Fields --> LogProb{use_rollout_logprobs or get_mismatch_metrics or use_tis}
    LogProb -- true --> AddLogProb[add rollout_log_probs]
    Fields --> Routing{use_rollout_routing_replay}
    Routing -- true --> AddRouting[add rollout_routed_experts]
    Fields --> MM{multimodal_keys is not None}
    MM -- true --> AddMM[add multimodal_train_inputs]
    Fields --> OPD{use_opd and opd_type == sglang}
    OPD -- true --> AddTeacher[add teacher_log_probs]
    Fields --> Extra[append transfer_queue_extra_data_fields]
```
