import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100



class TinyImageNetLabels:
    def __init__(self, root="tiny-imagenet-200"):
        with open(f"{root}/wnids.txt") as f:
            self.idx_to_wnid = [l.strip() for l in f]

        self.wnid_to_text = {}
        with open(f"{root}/words.txt") as f:
            for line in f:
                wnid, text = line.strip().split("\t")
                self.wnid_to_text[wnid] = text

        self.idx_to_text = [
            self.wnid_to_text[wnid] for wnid in self.idx_to_wnid
        ]

    def idx_to_label(self, idx):
        return self.idx_to_text[idx]



class TinyImageNetDataset(Dataset):
    """
    PyTorch Dataset for Tiny ImageNet classification.

    Args:
        root (str): path to tiny-imagenet-200
        split (str): 'train' | 'val' | 'test'
        transform (callable, optional): torchvision transforms
        return_wnid (bool): if True, also return WordNet ID
    """

    def __init__(self, root="tiny-imagenet-200", split="train", transform=None, return_wnid=False):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.transform = transform
        self.return_wnid = return_wnid

        # --------------------------------------------------
        # Load class list (WordNet IDs)
        # --------------------------------------------------
        wnids_path = os.path.join(root, "wnids.txt")
        with open(wnids_path, "r") as f:
            self.wnids = [line.strip() for line in f.readlines()]

        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        # --------------------------------------------------
        # Build samples list
        # --------------------------------------------------
        self.samples = []  # (image_path, class_idx, wnid)

        if split == "train":
            self._build_train()
        elif split == "val":
            self._build_val()
        else:  # test
            self._build_test()

    # --------------------------------------------------
    # Build splits
    # --------------------------------------------------
    def _build_train(self):
        train_dir = os.path.join(self.root, "train")

        for wnid in self.wnids:
            class_dir = os.path.join(train_dir, wnid, "images")
            if not os.path.isdir(class_dir):
                continue

            label = self.wnid_to_idx[wnid]

            for fname in os.listdir(class_dir):
                if fname.endswith(".JPEG"):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label, wnid))

    def _build_val(self):
        val_dir = os.path.join(self.root, "val")
        img_dir = os.path.join(val_dir, "images")
        ann_path = os.path.join(val_dir, "val_annotations.txt")

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name, wnid = parts[0], parts[1]

                label = self.wnid_to_idx[wnid]
                img_path = os.path.join(img_dir, img_name)

                self.samples.append((img_path, label, wnid))

    def _build_test(self):
        test_dir = os.path.join(self.root, "test", "images")

        for fname in os.listdir(test_dir):
            if fname.endswith(".JPEG"):
                path = os.path.join(test_dir, fname)
                self.samples.append((path, -1, None))  # no labels

    # --------------------------------------------------
    # Dataset API
    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, wnid = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.split == "test":
            return image, img_path  # no label

        if self.return_wnid:
            return image, label, wnid

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
            }