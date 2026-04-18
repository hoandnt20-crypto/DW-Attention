import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, path, rank):
    if rank != 0:
        return

    os.makedirs(path, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc
    }, os.path.join(path, "best_model.pth"))

def resume_checkpoint(model, optimizer, scheduler, path, device, rank):
    if not os.path.exists(path):
        return 0, 0.0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if rank == 0:
        print(f"Resume from epoch {checkpoint['epoch']}")

    return checkpoint["epoch"] + 1, checkpoint["val_acc"]