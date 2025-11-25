import os
from typing import Callable

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import pil_to_tensor

EDD_KEY_MAPPING = {"BE": 0, "cancer": 1, "HGD": 2, "polyp": 3, "suspicious": 4}


def generate_EDD_dataset_list(EDD_root: str):
    dataset_list = []
    image_list = os.listdir(os.path.join(EDD_root, "originalImages"))
    for image in image_list:
        image_path = image.split(".")[0]
        temp_label = []
        for key in EDD_KEY_MAPPING.keys():
            label_path = os.path.join(EDD_root, "masks", f"{image_path}_{key}.tif")
            if os.path.exists(label_path):
                temp_label.append(label_path)
        dataset_list.append(
            {
                "image": os.path.join(EDD_root, "originalImages", image),
                "label": temp_label,
            }
        )
    return dataset_list


class EDDDataset(data.Dataset):
    def __init__(
        self,
        samples: list,
        transform_img: Callable,
        transform_label: Callable,
        loader: Callable = default_loader,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.loader(sample["image"])

        label = torch.zeros((len(EDD_KEY_MAPPING), image.height, image.width))
        for label_path in sample["label"]:
            label_name = label_path.split("/")[-1].split(".")[0].split("_")[-1]
            label_img = pil_to_tensor(self.loader(label_path).convert("L"))
            mask = label_img == 255.0
            label[EDD_KEY_MAPPING[label_name]] = mask

        return self.transform_img(image), self.transform_label(label)


def give_augmentations(
    image_size: int = 352, image_mean: float = 0.5, image_std: float = 0.5
):
    train_transform_image = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    train_transform_label = transforms.Resize((image_size, image_size), antialias=True)
    val_transform_image = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    val_transform_label = transforms.Resize((image_size, image_size), antialias=True)
    return (
        train_transform_image,
        train_transform_label,
        val_transform_image,
        val_transform_label,
    )


def get_dataloader(config):
    data_root = config.dataset.EDD_seg.data_root
    train_ratio = config.dataset.EDD_seg.train_ratio
    dataset_list = generate_EDD_dataset_list(data_root)
    train_images = dataset_list[: int(len(dataset_list) * train_ratio)]
    val_images = dataset_list[int(len(dataset_list) * train_ratio) :]

    (
        train_transform_image,
        train_transform_label,
        val_transform_image,
        val_transform_label,
    ) = give_augmentations(
        image_size=config.dataset.EDD_seg.image_size, image_mean=config.dataset.EDD_seg.image_mean, image_std=config.dataset.EDD_seg.image_std
    )
    train_dataset = EDDDataset(
        train_images, train_transform_image, train_transform_label
    )
    val_dataset = EDDDataset(val_images, val_transform_image, val_transform_label)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.dataset.EDD_seg.batch_size,
        shuffle=True,
        num_workers=config.dataset.EDD_seg.num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=config.dataset.EDD_seg.batch_size,
        shuffle=False,
        num_workers=config.dataset.EDD_seg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    batch_size = 8
    num_workers = 4
    image_size = 352
    train_loader, test_loader = get_dataloader(
        "EDD2020", batch_size, num_workers, image_size=image_size
    )
    train_num = 0
    for i, (image, label) in enumerate(train_loader):
        print(image.size())
        print(label.size())
        train_num += 1
    test_num = 0
    for i, (image, label) in enumerate(test_loader):
        print(image.size())
        print(label.size())
        test_num += 1
    print(train_num)
    print(test_num)
    print(train_num + test_num)
