# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import optax
import tqdm
from flax.training import checkpoints, train_state


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from smc_clustering.diffusion.diffusion import VariationalDiffusion


def train_model(
    rng: jax.Array,
    model: VariationalDiffusion,
    dataloader: Iterable[tuple[jax.Array, jax.Array]],
    optimizer: optax.GradientTransformation | None = None,
    epochs: int = 1,
    loss_interval: int = 100,
    callback: Callable[..., object] | None = None,
    checkpoint_path: str | None = None,
) -> list[float]:
    loss_history = []

    if optimizer is None:
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(model.params)

    @jax.jit
    def update_step(
        rng: jax.Array, params: dict, x: jax.Array, masks: jax.Array, opt_state: optax.OptState
    ) -> tuple[jax.Array, dict, optax.OptState]:
        val, grads = jax.value_and_grad(model.loss, argnums=1, has_aux=False)(rng, params, x, masks)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return val, params, opt_state

    for epoch in range(1, epochs + 1):
        with tqdm.tqdm(dataloader) as pbar:
            for i, batch in enumerate(pbar):
                x, masks = batch
                rng, step_rng = jax.random.split(rng)
                loss, model.params, opt_state = update_step(
                    step_rng, model.params, x.numpy(), masks.numpy(), opt_state
                )

                if i % loss_interval == 0:
                    loss_history.append(loss.item())
                    pbar.set_description(f"epoch: {epoch}, loss: {loss.item():.4f}")
            if callback is not None:
                callback(model, loss_history)

        if checkpoint_path is not None and epoch % 5 == 0:
            state = train_state.TrainState.create(
                apply_fn=model.net.apply, params=model.params["params"], tx=optimizer
            )
            save_dict = {"model": state, "loss_history": loss_history}

            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_path, target=save_dict, step=epoch, overwrite=True, keep=2
            )

    model.compile_net()
    return loss_history
