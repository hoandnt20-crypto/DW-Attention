class TrainConfig:
    dataset      = "cifar10"
    epochs       = 300
    warmup_epochs= 20
    batch_size   = 128
    base_lr      = 5e-4
    min_lr       = 5e-6
    warmup_lr    = 5e-7
    wd           = 5e-2
    num_workers  = 4
    project_name = "cifar-10"
    data_root    = "./data"
    seed         = 39