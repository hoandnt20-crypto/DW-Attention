import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from configs.train_config import TrainConfig
from configs.model_config import ModelConfig

from utils.ddp import setup_ddp, cleanup_ddp
from utils.misc import set_seed
from utils.checkpoint import save_checkpoint, resume_checkpoint

from data.dataset import CIFARDataset
from transforms.transforms import get_transforms

from engine.train import train_one_epoch
from engine.eval import evaluate

from models.vit import ViT

cfg = TrainConfig()
model_cfg = ModelConfig()

def main():
    set_seed(cfg.seed)
    rank, world_size, local_rank = setup_ddp()
    device = f"cuda:{local_rank}"
    
    if rank == 0:
        wandb.login() # set up wandb key
    dist.barrier()  
    
    # Dataset
    train_ds = TinyImageNetDataset(
        root="tiny-imagenet-200",
        split="train",
        transform=get_transforms(train=True)
    )
    
    val_ds = TinyImageNetDataset(
        root="tiny-imagenet-200",
        split="val",
        transform=get_transforms(train=False)
    )

    # DDP Sampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoader
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        sampler=train_sampler, 
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )

    val_loader   = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        sampler=val_sampler, 
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )

    # Model
    model = ViT(
        image_size  = model_cfg.image_size,
        patch_size  = model_cfg.patch_size,
        num_classes = 200,
        dim         = model_cfg.dim,
        depth       = model_cfg.depth,
        heads       = model_cfg.heads,
        mlp_dim     = model_cfg.mlp_dim,
        dim_head    = model_cfg.dim_head,
        pool        = "mean",
        rdrop       = 0.4
    ).to(device)
    model = torch.compile(model)  # default mode
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer, Scheduler, Scaler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.wd)
    warmup_scheduler = LinearLR(optimizer, start_factor=cfg.warmup_lr/cfg.base_lr, total_iters=cfg.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[cfg.warmup_epochs])
    scaler = amp.GradScaler()

    # Loss fn
    val_criterion = nn.CrossEntropyLoss()
    train_criterion = SoftTargetCrossEntropy()

    # Only rank 0 logs
    if rank==0:
        run = wandb.init(
            project=cfg.project_name,
            name=f"{cfg.batch_size}-{cfg.base_lr}-{cfg.wd}",
            config={"dataset": cfg.dataset,"epochs": cfg.epochs,"batch_size": cfg.batch_size,"lr": cfg.base_lr,"model": "Efficient-ViT"}
        )
    else:
        run = None

    for epoch in range(start_epoch, cfg.epochs):
        if rank == 0:
            print(f"Epoch [{epoch+1}/{cfg.epochs}]")
        train_sampler.set_epoch(epoch)  # DDP sampler cần set mỗi epoch
        train_loss, train_acc, train_grad_norm  = train_one_epoch(model, train_loader, optimizer, scaler, train_criterion, device, world_size, rank)
        val_loss, val_acc     = evaluate(model, val_loader, val_criterion, device, world_size, rank)
    
        scheduler.step()
    
        if rank == 0:
            run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/grad_norm": train_grad_norm,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            print(f"Train Loss {train_loss:.4f}, Train Acc: {train_acc:.4f}, Grad norm: {train_grad_norm:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint nếu val_acc tốt hơn
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, scheduler, epoch, val_acc, path="checkpoints", name="best_model.pth", rank=rank)
            print("\n", "="*50, "\n")

            
    
    if run is not None:
        run.finish()
    cleanup_ddp()

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    main()