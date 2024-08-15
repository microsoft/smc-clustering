# Licensed under the MIT license.

import torch
import tqdm

from .schedule import LinearSchedule


def train_model(
    model,
    dataloader,
    optimizer=None,
    epochs=1,
    loss_interval=100,
    callback=None,
    schedule_optimizer=None,
    checkpoint_path=None,
    device='cuda'
):
    loss_history = []

    if optimizer is None:
        optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)

    do_schedule_opt = not isinstance(model.schedule, LinearSchedule)
    if do_schedule_opt and (schedule_optimizer is None):
        schedule_optimizer = torch.optim.Adam(model.schedule.parameters(), lr=1e-3)

    # @torch.compile TODO: figure out how to use this
    def step(x, masks):
        loss = model(x, masks)
        optimizer.zero_grad()
        if do_schedule_opt:
            schedule_optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # mutiply the gradients of the scheduler by 2 * loss
        # as per diffusion paper, https://arxiv.org/pdf/2107.00630.pdf
        # eq. 62
        if do_schedule_opt:
            for p in model.schedule.parameters():
                p.grad = p.grad * 2 * loss
            schedule_optimizer.step()

        return loss

    for epoch in range(epochs):
        with tqdm.tqdm(dataloader) as pbar:
            for i, batch in enumerate(pbar):
                x, masks = batch
                loss = step(x.to(device), masks.to(device))
                if i % loss_interval == 0:
                    loss_history.append(loss.item())
                    pbar.set_description(f'epoch: {epoch}, loss: {loss.item():.4f}')
            if callback is not None:
                callback(model, loss_history)

        if checkpoint_path is not None:
            save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss_history': loss_history}
            if do_schedule_opt:
                save_dict['schedule_optimizer'] = schedule_optimizer.state_dict()
            torch.save(save_dict, checkpoint_path)

    return loss_history, model
