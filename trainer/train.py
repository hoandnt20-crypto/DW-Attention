import torch
import tqdm
from torch import amp
from utils.ddp import reduce_tensor
from utils.misc import get_grad_norm


def train_one_epoch(model, loader, optimizer, scaler, criterion,
                    device, world_size, rank):

    model.train()
    total_loss = torch.tensor(0., device=device)
    total_correct = torch.tensor(0., device=device)
    total_samples = torch.tensor(0., device=device)
    total_grad_norm = torch.tensor(0., device=device)

    pbar = tqdm.tqdm(loader, desc="Training", disable=(rank != 0))

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if np.random.rand() < cfg.mix_prob:
            images, targets = mixup_cutmix(images, labels, cfg)
        else:
            targets = F.one_hot(labels, 200).float()

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs.float(), targets)

        # -----------------------
        # backward
        # -----------------------
        scaler.scale(loss).backward()

        # unscale để lấy grad thật
        scaler.unscale_(optimizer)

        # grad norm (scalar tensor)
        grad_norm = get_grad_norm(model)
        total_grad_norm += grad_norm

        # (optional) clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        # stats
        total_loss += loss.detach() * labels.size(0)
        total_correct += (outputs.argmax(1) == labels).sum()
        total_samples += labels.size(0)

    # -----------------------
    # DDP collector
    # -----------------------
    total_loss = reduce_tensor(total_loss, world_size)
    total_correct = reduce_tensor(total_correct, world_size)
    total_samples = reduce_tensor(total_samples, world_size)
    total_grad_norm = reduce_tensor(total_grad_norm, world_size)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_grad_norm = total_grad_norm / len(loader)

    return avg_loss.item(), avg_acc.item(), avg_grad_norm.item()