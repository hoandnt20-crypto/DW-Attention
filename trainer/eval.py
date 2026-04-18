import torch
from torch import amp
from utils.ddp import reduce_tensor

@torch.no_grad()
def evaluate(model, loader, criterion, device, world_size, rank):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = torch.tensor(0, device=device)

    pbar = tqdm.tqdm(loader, desc="Evaluating", disable=(rank != 0))

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.detach() * batch_size
        total_correct += (outputs.argmax(1) == labels).sum()
        total_samples += batch_size

    # Reduce loss và accuracy
    total_loss = reduce_tensor(total_loss, world_size)
    total_correct = reduce_tensor(total_correct, world_size)
    total_samples = reduce_tensor(total_samples, world_size)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Đồng bộ giữa các process trước khi kết thúc epoch
    dist.barrier()
    return avg_loss.item(), avg_acc.item()