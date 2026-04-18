import torch



def collate_fn(batch):
    image = []
    label = []
    for i in batch:
        image.append(i["image"])
        label.append(i["label"])
    
    image, label = [torch.stack(i, axis=0) for i in [image, label]]
    
    return {
        "image": image,
        "label": label
    }