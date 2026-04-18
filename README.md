# DW-Attention Vision Transformer (DDP Training)

## 🚀 Overview

Project này implement một Vision Transformer cải tiến với:

* Flash Attention (PyTorch 2.x)
* Depthwise Convolution trong Transformer block
* Distributed Data Parallel (DDP)
* Auto Mixed Precision (AMP): float16
* Warmup + Cosine LR Scheduler
* Torch Compile (tăng tốc runtime)

Phù hợp để:

* Training multi-GPU
* Benchmark kiến trúc ViT custom

---

## 📁 Project Structure (chi tiết)

```
dw_attention_ddp/
│
├── configs/
│   ├── train_config.py      # Hyperparameters training (lr, batch size, epochs...)
│   └── model_config.py      # Kiến trúc model (dim, depth, heads...)
│
├── data/
│   └── dataset.py           # Wrapper cho CIFAR10/100 → trả dict {"image","label"}
│
├── models/
│   ├── layers.py            # Các block nhỏ:
│   │                        #   - PreNorm
│   │                        #   - FeedForward
│   │                        #   - FlashAttention
│   │
│   └── vit.py               # Model chính:
│                            #   - Transformer (DW-Conv + Attention)
│                            #   - ViT (patch embedding + cls token)
│
├── engine/
│   ├── train.py             # train_one_epoch():
│   │                        #   - forward + backward
│   │                        #   - AMP + GradScaler
│   │                        #   - tính accuracy + grad norm
│   │
│   └── eval.py              # evaluate():
│                            #   - validation loop
│                            #   - reduce metric across GPUs
│
├── transforms/
│   └── transforms.py        # Data augmentation:
│                            #   - RandomResizedCrop
│                            #   - AutoAugment
│                            #   - Normalize
│
├── utils/
│   ├── ddp.py               # DDP utilities:
│   │                        #   - setup_ddp()
│   │                        #   - cleanup_ddp()
│   │                        #   - reduce_tensor()
│   │
│   ├── misc.py              # Utilities:
│   │                        #   - set_seed()
│   │                        #   - grad norm
│   │
│   └── checkpoint.py        # Save / Resume training:
│                            #   - save_checkpoint()
│                            #   - resume_checkpoint()
│
├── main.py                  # Entry point:
│                            #   - build dataset
│                            #   - build model
│                            #   - torch.compile
│                            #   - training loop
│
└── README.md
```

---

## ⚙️ Installation

```bash
pip install torch torchvision einops wandb
```

Yêu cầu:

* PyTorch >= 2.0
* CUDA >= 11.7

---

## 🧠 Training (CLI)

### 🔹 Multi-GPU (DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:1222 --nnodes 1 --nproc-per-node 2 main.py
```

---

## ⚡ Training Pipeline (flow)

1. Setup DDP (rank, world_size)
2. Load dataset + DistributedSampler
3. Build ViT model
4. `torch.compile(model)` → tối ưu graph
5. Wrap với `DistributedDataParallel`
6. Optimizer: AdamW
7. Scheduler:

   * Warmup (LinearLR)
   * CosineAnnealing
8. Training loop:

   * AMP forward/backward
   * GradScaler
   * Reduce metrics across GPUs
9. Save best checkpoint

---

## 🔥 Key Design

### 1. Flash Attention

```python
F.scaled_dot_product_attention(...)
```

→ nhanh hơn attention truyền thống

---

### 2. Depthwise Conv trong Transformer

* Inject spatial inductive bias
* Giống ConvNet + Transformer hybrid

---

### 3. Torch Compile

```python
model = torch.compile(model)
```

→ speedup ~10–30%

---

### 4. DDP

* Scale lên nhiều GPU
* Sync gradient tự động

---

## 📊 Metrics

Training log:

* Loss
* Accuracy
* Gradient norm
* Learning rate

---

## 💾 Checkpoint

Auto save khi val acc tốt hơn:

```
checkpoints/best_model.pth
```

Resume tự động nếu tồn tại.

---
