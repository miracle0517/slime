from argparse import Namespace

import pytest

from slime.utils.transfer_queue import (
    add_total_lengths,
    actor_train_data_fields,
    critic_values_via_transfer_queue,
    default_train_data_fields,
    dict_to_tensordict,
    normalize_train_data_for_transfer_queue,
    transfer_queue_data_parallel_size,
)


def _args(**overrides):
    defaults = dict(
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        use_tis=False,
        use_rollout_routing_replay=False,
        multimodal_keys=None,
        use_opd=False,
        opd_type=None,
        use_critic=False,
        use_transfer_queue=False,
        transfer_queue_extra_data_fields=[],
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.unit
def test_default_train_data_fields_only_requests_enabled_optional_fields():
    fields = default_train_data_fields(
        _args(
            use_rollout_logprobs=True,
            use_rollout_routing_replay=True,
            multimodal_keys={"image": "image"},
            use_opd=True,
            opd_type="sglang",
            transfer_queue_extra_data_fields=["metadata"],
        )
    )

    assert "tokens" in fields
    assert "rollout_log_probs" in fields
    assert "rollout_routed_experts" in fields
    assert "multimodal_train_inputs" in fields
    assert "teacher_log_probs" in fields
    assert "metadata" in fields


@pytest.mark.unit
def test_default_train_data_fields_skips_absent_optional_fields():
    fields = default_train_data_fields(_args())

    assert "tokens" in fields
    assert "rollout_log_probs" not in fields
    assert "rollout_routed_experts" not in fields
    assert "multimodal_train_inputs" not in fields
    assert "teacher_log_probs" not in fields


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["get_mismatch_metrics", "use_tis"])
def test_default_train_data_fields_requests_rollout_log_probs_for_mismatch_modes(flag):
    fields = default_train_data_fields(_args(**{flag: True}))

    assert "rollout_log_probs" in fields


@pytest.mark.unit
def test_actor_train_data_fields_requests_values_when_critic_writeback_is_supported():
    fields = actor_train_data_fields(_args(use_transfer_queue=True, use_critic=True, context_parallel_size=1))

    assert critic_values_via_transfer_queue(_args(use_transfer_queue=True, use_critic=True, context_parallel_size=1))
    assert "values" in fields


@pytest.mark.unit
def test_actor_train_data_fields_skips_values_when_context_parallel_needs_rank_local_values():
    fields = actor_train_data_fields(_args(use_transfer_queue=True, use_critic=True, context_parallel_size=2))

    assert not critic_values_via_transfer_queue(_args(use_transfer_queue=True, use_critic=True, context_parallel_size=2))
    assert "values" not in fields


@pytest.mark.unit
def test_add_total_lengths_derives_lengths_without_mutating_input():
    data = {"tokens": [[1, 2, 3], [4, 5]], "rewards": [1.0, 0.0]}

    output = add_total_lengths(data)

    assert output["total_lengths"] == [3, 2]
    assert "total_lengths" not in data


@pytest.mark.unit
def test_normalize_train_data_for_transfer_queue_fills_requested_base_fields():
    data = {
        "tokens": [[1, 2, 3], [4, 5]],
        "response_lengths": [2, 1],
        "loss_masks": [[1, 1], [1]],
        "rewards": [1.0, 0.0],
    }

    output = normalize_train_data_for_transfer_queue(data)

    assert output["total_lengths"] == [3, 2]
    assert output["raw_reward"] == [1.0, 0.0]
    assert output["truncated"] == [0, 0]
    assert output["sample_indices"] == [0, 1]


@pytest.mark.unit
def test_transfer_queue_data_parallel_size_excludes_model_parallel_ranks():
    assert transfer_queue_data_parallel_size(_args()) == 2


@pytest.mark.unit
def test_dict_to_tensordict_converts_jagged_rollout_fields():
    pytest.importorskip("tensordict")

    td = dict_to_tensordict(
        {
            "tokens": [[1, 2, 3], [4, 5]],
            "loss_masks": [[1, 1], [1]],
            "response_lengths": [2, 1],
            "rewards": [1.0, 0.0],
        },
        batch_size=2,
    )

    assert td.batch_size.numel() == 2
    assert td["response_lengths"].tolist() == [2, 1]
    assert "tokens" in td.keys()


@pytest.mark.unit
def test_dict_to_tensordict_converts_tensor_list_fields_for_writeback():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")

    td = dict_to_tensordict(
        {
            "values": [
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.3]),
            ],
        },
        batch_size=2,
    )

    assert "values" in td.keys()
