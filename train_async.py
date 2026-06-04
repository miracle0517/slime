import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.transfer_queue import critic_values_via_transfer_queue, initialize_transfer_queue, transfer_queue_enabled

logger = logging.getLogger(__name__)


def _transfer_queue_prefetch_window(args) -> int:
    return max(1, int(getattr(args, "max_staleness", 0)) + 1)


def _fill_transfer_queue_rollout_window(
    args,
    rollout_manager,
    pending_rollouts: dict[int, ray.ObjectRef],
    next_rollout_id: int,
) -> int:
    window = _transfer_queue_prefetch_window(args)
    while len(pending_rollouts) < window and next_rollout_id < args.num_rollout:
        pending_rollouts[next_rollout_id] = rollout_manager.generate.remote(next_rollout_id)
        logger.info(
            "Scheduled TransferQueue async rollout: rollout_id=%s pending=%s window=%s",
            next_rollout_id,
            sorted(pending_rollouts),
            window,
        )
        next_rollout_id += 1
    return next_rollout_id


def _wait_training_refs_with_rollout_write(
    rollout_id: int,
    train_refs: list[ray.ObjectRef],
    rollout_write_ref: ray.ObjectRef | None,
) -> None:
    """Wait for train refs while surfacing rollout write failures early."""
    pending_train_refs = list(train_refs)
    rollout_write_pending = rollout_write_ref is not None

    while pending_train_refs:
        wait_refs = list(pending_train_refs)
        if rollout_write_pending:
            wait_refs.append(rollout_write_ref)

        ready_refs, _ = ray.wait(wait_refs, num_returns=1)
        for ref in ready_refs:
            if rollout_write_pending and ref == rollout_write_ref:
                ray.get(ref)
                rollout_write_pending = False
                logger.info("TransferQueue rollout write completed: rollout_id=%s", rollout_id)
            else:
                ray.get(ref)
                pending_train_refs.remove(ref)

    if rollout_write_pending:
        ray.get(rollout_write_ref)
        logger.info("TransferQueue rollout write completed: rollout_id=%s", rollout_id)


def _wait_all_transfer_queue_rollouts(pending_rollouts: dict[int, ray.ObjectRef]) -> None:
    if not pending_rollouts:
        return
    logger.info("Waiting for pending TransferQueue rollouts before weight update: %s", sorted(pending_rollouts))
    ray.get(list(pending_rollouts.values()))


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    initialize_transfer_queue(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    use_transfer_queue = transfer_queue_enabled(args)
    if use_transfer_queue:
        pending_transfer_queue_rollouts: dict[int, ray.ObjectRef] = {}
        next_transfer_queue_rollout_id = args.start_rollout_id
        next_transfer_queue_rollout_id = _fill_transfer_queue_rollout_window(
            args,
            rollout_manager,
            pending_transfer_queue_rollouts,
            next_transfer_queue_rollout_id,
        )
        rollout_data_next_future = None
    else:
        pending_transfer_queue_rollouts = {}
        rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    # async train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_write_ref = None
        if use_transfer_queue:
            rollout_data_curr_ref = None
            rollout_write_ref = pending_transfer_queue_rollouts.pop(rollout_id, None)
            if rollout_write_ref is None:
                rollout_write_ref = rollout_manager.generate.remote(rollout_id)
            next_transfer_queue_rollout_id = _fill_transfer_queue_rollout_window(
                args,
                rollout_manager,
                pending_transfer_queue_rollouts,
                next_transfer_queue_rollout_id,
            )
        else:
            # Sync the last generation
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)

            # Start the next rollout early.
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                if critic_values_via_transfer_queue(args):
                    train_refs = actor_model.async_train(rollout_id, rollout_data_curr_ref)
                    _wait_training_refs_with_rollout_write(rollout_id, value_refs + train_refs, rollout_write_ref)
                elif use_transfer_queue:
                    train_refs = actor_model.async_train(
                        rollout_id,
                        rollout_data_curr_ref,
                        external_data=value_refs,
                    )
                    _wait_training_refs_with_rollout_write(rollout_id, value_refs + train_refs, rollout_write_ref)
                else:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                if use_transfer_queue:
                    _wait_training_refs_with_rollout_write(rollout_id, value_refs, rollout_write_ref)
                    ray.get(rollout_manager.clear_transfer_queue_partition.remote(rollout_id))
                else:
                    ray.get(value_refs)
                    if args.use_transfer_queue:
                        ray.get(rollout_manager.clear_transfer_queue_partition.remote(rollout_id))
        else:
            train_refs = actor_model.async_train(rollout_id, rollout_data_curr_ref)
            if use_transfer_queue:
                _wait_training_refs_with_rollout_write(rollout_id, train_refs, rollout_write_ref)
            else:
                ray.get(train_refs)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            if use_transfer_queue:
                # Rollout engines must not receive weight updates in the middle of a generation call.
                _wait_all_transfer_queue_rollouts(pending_transfer_queue_rollouts)
            else:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            actor_model.update_weights()
            if use_transfer_queue:
                next_transfer_queue_rollout_id = _fill_transfer_queue_rollout_window(
                    args,
                    rollout_manager,
                    pending_transfer_queue_rollouts,
                    next_transfer_queue_rollout_id,
                )

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
