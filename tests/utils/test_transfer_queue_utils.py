from argparse import Namespace

import pytest

from slime.utils.transfer_queue import add_total_lengths, default_train_data_fields, dict_to_tensordict


def _args(**overrides):
    defaults = dict(
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        use_tis=False,
        use_rollout_routing_replay=False,
        multimodal_keys=None,
        use_opd=False,
        opd_type=None,
        transfer_queue_extra_data_fields=[],
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
def test_add_total_lengths_derives_lengths_without_mutating_input():
    data = {"tokens": [[1, 2, 3], [4, 5]], "rewards": [1.0, 0.0]}

    output = add_total_lengths(data)

    assert output["total_lengths"] == [3, 2]
    assert "total_lengths" not in data


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
