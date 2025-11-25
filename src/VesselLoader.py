import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Sampler, BatchSampler
import numpy as np
from PIL import Image
import random
from easydict import EasyDict
from typing import Callable, Tuple, List, Dict, Optional, Union, Iterator
import math


class Colors:
    BLACK = "\033[0;30m";
    RED = "\033[0;31m";
    GREEN = "\033[0;32m";
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m";
    PURPLE = "\033[0;35m";
    CYAN = "\033[0;36m";
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m";
    LIGHT_RED = "\033[1;31m";
    LIGHT_GREEN = "\033[1;32m";
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m";
    LIGHT_PURPLE = "\033[1;35m";
    LIGHT_CYAN = "\033[1;36m";
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m";
    FAINT = "\033[2m";
    ITALIC = "\033[3m";
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m";
    NEGATIVE = "\033[7m";
    CROSSED = "\033[9m";
    END = "\033[0m"


def cut_mix(_input: Image.Image, mask_1: Image.Image, _refer: Image.Image, mask_2: Image.Image) -> Tuple[
    Image.Image, Image.Image]:
    random_gen = random.Random()
    _input_np = np.array(_input)
    mask_1_np = np.array(mask_1)
    _refer_np = np.array(_refer)
    mask_2_np = np.array(mask_2)

    h1, w1 = _input_np.shape[:2]
    h2, w2 = _refer_np.shape[:2]

    is_mask1_rgb = len(mask_1_np.shape) == 3 and mask_1_np.shape[2] == 3
    is_mask2_rgb = len(mask_2_np.shape) == 3 and mask_2_np.shape[2] == 3

    rand_x = random_gen.random() * 0.75
    rand_y = random_gen.random() * 0.75
    rand_w = random_gen.random() * 0.5
    rand_h = random_gen.random() * 0.5

    cw_1 = int((rand_w + 0.25) * w1)
    ch_1 = int((rand_h + 0.25) * h1)

    cx_1 = int(rand_x * (w1 - cw_1))
    cy_1 = int(rand_y * (h1 - ch_1))

    cw_2 = int((rand_w + 0.25) * w2)
    ch_2 = int((rand_h + 0.25) * h2)

    cx_2 = int(rand_x * (w2 - cw_2))
    cy_2 = int(rand_y * (h2 - ch_2))

    if ch_2 == 0 or cw_2 == 0 or ch_1 == 0 or cw_1 == 0:
        return Image.fromarray(_input_np.astype(np.uint8)), Image.fromarray(mask_1_np.astype(np.uint8))

    cutout_img_src = _refer_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]

    if is_mask2_rgb:
        cutout_mask_src = mask_2_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2, :]
    else:
        cutout_mask_src = mask_2_np[cy_2:cy_2 + ch_2, cx_2:cx_2 + cw_2]

    if cutout_img_src.size == 0 or cutout_mask_src.size == 0:
        return Image.fromarray(_input_np.astype(np.uint8)), Image.fromarray(mask_1_np.astype(np.uint8))

    import cv2
    cutout_img_resized = cv2.resize(cutout_img_src, (cw_1, ch_1), interpolation=cv2.INTER_LINEAR)
    cutout_mask_resized = cv2.resize(cutout_mask_src, (cw_1, ch_1), interpolation=cv2.INTER_NEAREST)

    if not is_mask2_rgb and len(cutout_mask_resized.shape) == 3:
        cutout_mask_resized = cutout_mask_resized.squeeze(-1)

    _input_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_img_resized

    if is_mask1_rgb:
        mask_1_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1, :] = cutout_mask_resized
    else:
        mask_1_np[cy_1:cy_1 + ch_1, cx_1:cx_1 + cw_1] = cutout_mask_resized

    return Image.fromarray(_input_np.astype(np.uint8)), Image.fromarray(mask_1_np.astype(np.uint8))


def center_padding(img: Union[torch.Tensor, Image.Image], target_size: List[int], pad_digit: int = 0) -> Union[
    torch.Tensor, Image.Image]:
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        original_mode = img.mode
        img_tensor = TF.to_tensor(img)
    else:
        img_tensor = img

    if img_tensor.ndim == 4:
        in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    elif img_tensor.ndim == 3:
        in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    elif img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)
        in_h, in_w = img_tensor.shape[-2], img_tensor.shape[-1]
    else:
        raise ValueError(f"Unsupported tensor ndim: {img_tensor.ndim}")

    target_h, target_w = target_size[0], target_size[1]

    if in_h >= target_h and in_w >= target_w:
        return img

    pad_left = max(0, (target_w - in_w) // 2)
    pad_right = max(0, target_w - in_w - pad_left)
    pad_top = max(0, (target_h - in_h) // 2)
    pad_bot = max(0, target_h - in_h - pad_top)

    import torch.nn.functional as F
    tensor_padded = F.pad(img_tensor, [pad_left, pad_right, pad_top, pad_bot], 'constant', pad_digit)

    if is_pil:
        pil_image_padded = TF.to_pil_image(
            tensor_padded.squeeze(0) if tensor_padded.ndim == 4 and tensor_padded.shape[0] == 1 else tensor_padded,
            mode=original_mode if original_mode in ['L', 'RGB'] else None)
        return pil_image_padded
    else:
        return tensor_padded


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _RepeatSampler(Sampler[int]):

    def __init__(self, index_sampler: Sampler[int]):
        self.index_sampler = index_sampler

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from iter(self.index_sampler)

    def __len__(self) -> int:
        return len(self.index_sampler)

class MultiEpochsDataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (isinstance(self.dataset, torch.utils.data.IterableDataset)):
            if self.batch_sampler is None:
                raise RuntimeError(
                    "MultiEpochsDataLoader: DataLoader.batch_sampler was not initialized by super().__init__ for a map-style dataset.")

            self._len_one_epoch = len(self.batch_sampler)
            original_index_sampler = self.batch_sampler.sampler
            repeated_index_sampler = _RepeatSampler(original_index_sampler)
            self.batch_sampler.sampler = repeated_index_sampler

        elif isinstance(self.dataset, torch.utils.data.IterableDataset):
            print(f"{Colors.YELLOW}Warning: MultiEpochsDataLoader used with an IterableDataset. "
                  f"Repeating behavior and length determination might not be as expected. "
                  f"Ensure your IterableDataset handles iteration as desired.{Colors.END}")
            if hasattr(self.dataset, '__len__') and self.batch_size is not None:
                self._len_one_epoch = math.ceil(len(self.dataset) / self.batch_size)
            else:
                raise ValueError("MultiEpochsDataLoader: Cannot determine length for this IterableDataset "
                                 "configuration to define one epoch. Please ensure it has a __len__ "
                                 "or use a different loader for unbounded iterable datasets.")
        else:
            raise TypeError("Unsupported dataset type for MultiEpochsDataLoader.")

        self.persistent_iterator = super().__iter__()

    def __len__(self) -> int:
        return self._len_one_epoch

    def __iter__(self) -> Iterator:
        for _ in range(self._len_one_epoch):
            yield next(self.persistent_iterator)


def generate_dataset_list(
        phase_root: str,
        image_subdir: str,
        label_subdir: str,
        label_filename_pattern: str,
) -> List[Dict[str, str]]:
    dataset_list = []
    image_folder_path = os.path.join(phase_root, image_subdir)
    label_folder_path = os.path.join(phase_root, label_subdir)

    if not os.path.isdir(image_folder_path):
        print(f"{Colors.RED}Warning: Image folder not found: {image_folder_path}{Colors.END}")
        return dataset_list
    if not os.path.isdir(label_folder_path):
        print(f"{Colors.RED}Warning: Label folder not found: {label_folder_path}{Colors.END}")
        return dataset_list

    image_files = sorted(os.listdir(image_folder_path))

    for img_filename in image_files:
        image_base_name, _ = os.path.splitext(img_filename)
        expected_label_filename = label_filename_pattern.format(base_name=image_base_name)
        full_label_path = os.path.join(label_folder_path, expected_label_filename)
        full_image_path = os.path.join(image_folder_path, img_filename)

        if os.path.exists(full_label_path) and os.path.exists(full_image_path):
            dataset_list.append({"image": full_image_path, "label": full_label_path})
        elif not os.path.exists(full_label_path):
            print(
                f"{Colors.YELLOW}Warning: No corresponding label found for image {img_filename}. Looked for {full_label_path}{Colors.END}")
        elif not os.path.exists(full_image_path):
            print(f"{Colors.YELLOW}Warning: Image file somehow not found: {full_image_path}{Colors.END}")
    return dataset_list


class _VesselDatasetInternal(Dataset):

    def __init__(
        self,
        samples: List[Dict[str, str]],
        mode: str,
        dataset_config: EasyDict,
        loader: Callable = Image.open,
    ):
        super().__init__()
        self.samples = samples
        self.mode = mode
        self.args = dataset_config
        self.loader = loader

        if isinstance(self.args.image_size, int):
            self.args.image_size = [self.args.image_size, self.args.image_size]

        self.image_mean = self.args.image_mean
        self.image_std = self.args.image_std

        self.img_paths_x = [s["image"] for s in self.samples]
        self.img_paths_y = [s["label"] for s in self.samples]

        print(
            f'{Colors.LIGHT_RED}Mounting data on memory... Dataset: {self.args.get("name", "N/A")}, Mode: {self.mode}{Colors.END}')
        self.images_pil_x = []
        self.images_pil_y = []
        for idx in range(len(self.img_paths_x)):
            try:
                img_x = self.loader(self.img_paths_x[idx]).convert('RGB')
                img_y = self.loader(self.img_paths_y[idx]).convert('L')
                self.images_pil_x.append(img_x)
                self.images_pil_y.append(img_y)
            except Exception as e:
                print(
                    f"{Colors.RED}Error loading image/label pair: {self.img_paths_x[idx]} or {self.img_paths_y[idx]}. Error: {e}{Colors.END}")
                raise
        if not self.images_pil_x:
            print(
                f"{Colors.RED}Warning: No images were loaded for mode {self.mode}. Please check paths and data.{Colors.END}")

    def __len__(self) -> int:
        return len(self.images_pil_x)

    def _transform(self, image: Image.Image, target: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_p = image.copy()
        target_p = target.copy()
        target_h, target_w = self.args.image_size[0], self.args.image_size[1]

        if self.mode == 'validation' or self.mode == 'test':
            img_w_orig, img_h_orig = image_p.size
            if img_h_orig < target_h or img_w_orig < target_w:
                image_p = center_padding(image_p, [target_h, target_w], pad_digit=0)
                target_p = center_padding(target_p, [target_h, target_w], pad_digit=0)


        if self.mode == 'train':
            if torch.rand(1).item() > 0.5:
                image_p = TF.hflip(image_p)
                target_p = TF.hflip(target_p)
            if torch.rand(1).item() > 0.5:
                image_p = TF.vflip(image_p)
                target_p = TF.vflip(target_p)

            if hasattr(self.args, 'transform_cutmix') and self.args.transform_cutmix:
                if random.random() < self.args.get('transform_cutmix_prob', 0.5):
                    rand_idx = random.randint(0, len(self.images_pil_x) - 1)
                    ref_img = self.images_pil_x[rand_idx].copy()
                    ref_lbl = self.images_pil_y[rand_idx].copy()
                    image_p, target_p = cut_mix(image_p, target_p, ref_img, ref_lbl)


            if hasattr(self.args, 'transform_random_resized_crop') and self.args.transform_random_resized_crop:
                if random.random() < self.args.get('transform_random_resized_crop_prob', 0.5):
                    scale = self.args.get('transform_random_resized_crop_scale', (0.5, 1.5))
                    i, j, h, w = transforms.RandomResizedCrop.get_params(image_p, scale=scale, ratio=(0.75, 1.33))
                    image_p = TF.resized_crop(image_p, i, j, h, w, [target_h, target_w], antialias=True)
                    target_p = TF.resized_crop(target_p, i, j, h, w, [target_h, target_w],
                                               interpolation=InterpolationMode.NEAREST)

        img_transform_list = [
            transforms.Resize([target_h, target_w], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ]
        if self.mode == 'train':
            if hasattr(self.args, 'transform_jitter') and self.args.transform_jitter:
                if random.random() < self.args.get('transform_jitter_prob', 0.8):
                    jitter_params = self.args.get('jitter_params',
                                                  {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1})
                    img_transform_list.insert(0, transforms.ColorJitter(**jitter_params)) # Jitter PIL image
            if hasattr(self.args, 'transform_blur') and self.args.transform_blur:
                if random.random() < self.args.get('transform_blur_prob', 0.5):
                    kernel_size = int((random.random() * 8 + 3).__round__())
                    if kernel_size % 2 == 0: kernel_size = max(3, kernel_size - 1)
                    else: kernel_size = max(3, kernel_size)
                    blur_sigma = self.args.get('blur_sigma', (0.1, 2.0))
                    img_transform_list.insert(0, transforms.GaussianBlur(kernel_size=kernel_size, sigma=blur_sigma))


        image_tensor = transforms.Compose(img_transform_list)(image_p)

        if target_p.mode != 'L':
            target_p = target_p.convert('L')
        lbl_tensor_raw = TF.to_tensor(target_p)
        lbl_tensor_binary = (lbl_tensor_raw > 0.5).float()
        target_tensor = TF.resize(lbl_tensor_binary, [target_h, target_w],
                                  interpolation=InterpolationMode.NEAREST, antialias=False)
        return image_tensor, target_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        if index >= len(self.images_pil_x):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.images_pil_x)}")

        img_x_pil = self.images_pil_x[index]
        img_y_pil = self.images_pil_y[index]

        img_x_tensor, img_y_tensor = self._transform(img_x_pil, img_y_pil)
        return img_x_tensor, img_y_tensor, self.img_paths_x[index], self.img_paths_y[index]


class Image2ImageLoader_zero_pad:
    def __init__(self,
                 samples: List[Dict[str, str]],
                 mode: str,
                 dataset_config: EasyDict,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 ):
        self.dataset_config = dataset_config
        self.mode = mode
        self.image_loader_dataset = _VesselDatasetInternal(
            samples=samples,
            mode=mode,
            dataset_config=dataset_config
        )
        g = torch.Generator()
        g.manual_seed(self.dataset_config.get('random_seed', 3407))

        self.Loader = TorchDataLoader(
            self.image_loader_dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            worker_init_fn=seed_worker, # seed_worker is still useful
            generator=g,
            pin_memory=pin_memory,
            drop_last=(mode == 'train')
        )

    def __len__(self):
        return len(self.image_loader_dataset)



def get_dataloader(config: EasyDict) -> Tuple[
    Optional[TorchDataLoader],
    Optional[TorchDataLoader],
    # Optional[TorchDataLoader],
]:
    dataset_chosen_name = config.trainer.dataset_choose
    try:
        dataset_params = config.dataset[dataset_chosen_name]
        dataset_params.name = dataset_chosen_name
    except KeyError:
        raise ValueError(
            f"Dataset '{dataset_chosen_name}' not found in config.dataset. Available: {list(config.dataset.keys())}")

    data_root = dataset_params.data_root
    batch_size = dataset_params.batch_size
    num_workers = dataset_params.num_workers

    cfg_train_dir = getattr(dataset_params, "train_dir", "train")
    cfg_val_dir = getattr(dataset_params, "val_dir", "val")
    cfg_test_dir = getattr(dataset_params, "test_dir", None)
    cfg_image_subdir = getattr(dataset_params, "image_subdir", "input")
    cfg_label_subdir = getattr(dataset_params, "label_subdir", "label")
    # DRIVE
    cfg_train_label_pattern = getattr(dataset_params, "train_label_pattern", "{base_name}.png")
    cfg_val_label_pattern = getattr(dataset_params, "val_label_pattern", "{base_name}_manual1.png")
    # STARE
    # cfg_train_label_pattern = getattr(dataset_params, "train_label_pattern", "{base_name}.ah.ppm")
    # cfg_val_label_pattern = getattr(dataset_params, "val_label_pattern", "{base_name}.ah.ppm")
    cfg_test_label_pattern = getattr(dataset_params, "test_label_pattern", cfg_val_label_pattern)
    # cfg_test_label_pattern = getattr(dataset_params, "test_label_pattern", cfg_val_label_pattern)

    train_loader, val_loader, test_loader = None, None, None

    train_phase_root = os.path.join(data_root, cfg_train_dir)
    if os.path.isdir(train_phase_root):
        train_samples = generate_dataset_list(
            phase_root=train_phase_root, image_subdir=cfg_image_subdir,
            label_subdir=cfg_label_subdir, label_filename_pattern=cfg_train_label_pattern
        )
        if train_samples:
            train_dataloader_wrapper = Image2ImageLoader_zero_pad(
                samples=train_samples, mode='train', dataset_config=dataset_params,
                batch_size=batch_size, num_workers=num_workers
            )
            train_loader = train_dataloader_wrapper.Loader
            print(f"{Colors.GREEN}Loaded {len(train_samples)} training samples from {train_phase_root}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}No training samples found in {train_phase_root}{Colors.END}")
    else:
        print(f"{Colors.RED}Training directory not found: {train_phase_root}{Colors.END}")

    val_phase_root = os.path.join(data_root, cfg_val_dir)
    if os.path.isdir(val_phase_root):
        val_samples = generate_dataset_list(
            phase_root=val_phase_root, image_subdir=cfg_image_subdir,
            label_subdir=cfg_label_subdir, label_filename_pattern=cfg_val_label_pattern
        )
        if val_samples:
            val_dataloader_wrapper = Image2ImageLoader_zero_pad(
                samples=val_samples, mode='validation', dataset_config=dataset_params,
                batch_size=batch_size, num_workers=num_workers
            )
            val_loader = val_dataloader_wrapper.Loader
            print(f"{Colors.GREEN}Loaded {len(val_samples)} validation samples from {val_phase_root}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}No validation samples found in {val_phase_root}{Colors.END}")
    else:
        print(f"{Colors.RED}Validation directory not found: {val_phase_root}{Colors.END}")

    if cfg_test_dir:
        test_phase_root = os.path.join(data_root, cfg_test_dir)
        if os.path.isdir(test_phase_root):
            test_samples = generate_dataset_list(
                phase_root=test_phase_root, image_subdir=cfg_image_subdir,
                label_subdir=cfg_label_subdir, label_filename_pattern=cfg_test_label_pattern
            )
            if test_samples:
                test_dataloader_wrapper = Image2ImageLoader_zero_pad(
                    samples=test_samples, mode='test', dataset_config=dataset_params,
                    batch_size=batch_size, num_workers=num_workers
                )
                test_loader = test_dataloader_wrapper.Loader
                print(f"{Colors.GREEN}Loaded {len(test_samples)} test samples from {test_phase_root}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}No test samples found in {test_phase_root}{Colors.END}")
        else:
            print(f"{Colors.RED}Test directory specified but not found: {test_phase_root}{Colors.END}")
    else:
        print(f"{Colors.BLUE}No test directory specified in config.{Colors.END}")

    return train_loader, val_loader # test_loader
