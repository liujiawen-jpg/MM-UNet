import os
from typing import Callable

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import pil_to_tensor

def generate_Train_dataset_list(dataset_root: str):
    dataset_list = []
    imagepath = os.path.join(dataset_root, "TrainDataset", "Frame")
    labelpath = os.path.join(dataset_root, "TrainDataset", "GT")
    for folder in os.listdir(imagepath):
        image_path = os.path.join(imagepath, folder)
        label_path = os.path.join(labelpath, folder)
        for image in os.listdir(image_path):
            image_name = image.split(".")[0]
            dataset_list.append(
            {
                "image": os.path.join(image_path, image),
                "label": os.path.join(label_path, image_name + ".png"),
            }
        )
    return dataset_list

def generate_Test_dataset_list(dataset_root: str):
    dataset_list = []
    imagepath = os.path.join(dataset_root, "TestHardDataset", "Unseen", "Frame")
    labelpath = os.path.join(dataset_root, "TestHardDataset", "Unseen", "GT")
    for folder in os.listdir(imagepath):
        image_path = os.path.join(imagepath, folder)
        label_path = os.path.join(labelpath, folder)
        for image in os.listdir(image_path):
            image_name = image.split(".")[0]
            dataset_list.append(
            {
                "image": os.path.join(image_path, image),
                "label": os.path.join(label_path, image_name + ".png"),
            }
        )
    return dataset_list

class SunSegDataset(data.Dataset):
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
        label_img = pil_to_tensor(self.loader(sample["label"]).convert("L"))
        label = torch.zeros(image.height, image.width)
        label = label_img == 255.0
        

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
    data_root = config.dataset.Sun_seg.data_root
    train_images = generate_Train_dataset_list(data_root)
    val_images = generate_Test_dataset_list(data_root)
    (
        train_transform_image,
        train_transform_label,
        val_transform_image,
        val_transform_label,
    ) = give_augmentations(
        image_size=config.dataset.Sun_seg.image_size, image_mean=config.dataset.Sun_seg.image_mean, image_std=config.dataset.Sun_seg.image_std
    )
    train_dataset = SunSegDataset(
        train_images, train_transform_image, train_transform_label
    )
    val_dataset = SunSegDataset(val_images, val_transform_image, val_transform_label)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.dataset.Sun_seg.batch_size,
        shuffle=True,
        num_workers=config.dataset.Sun_seg.num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=config.dataset.Sun_seg.batch_size,
        shuffle=False,
        num_workers=config.dataset.Sun_seg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader



if __name__ == "__main__":
    data_root = "/root/.cache/huggingface/forget/datasets/SUN Colonscopy Video/data/SUN-SEG/"
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