import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)


def ImageTransform(loadSize):
    return {
        "train": Compose(
            [
                Resize(loadSize, interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
            ]
        ),
        "test": Compose(
            [
                Resize(loadSize, interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
            ]
        ),
    }


class UIEData(Dataset):
    def __init__(
        self, path_img, path_gt, path_gt_depth, path_img_hist, loadSize, mode=1
    ):
        super().__init__()
        self.path_img = path_img
        self.path_gt = path_gt
        self.path_gt_depth = path_gt_depth
        self.path_img_hist = path_img_hist

        self.loadsize = loadSize  # e.g. (336, 336) or 336
        self.crop_pad_size = (
            loadSize[0] if isinstance(loadSize, (tuple, list)) else loadSize
        )
        self.mode = mode

        self.data_img = os.listdir(self.path_img)
        self.data_gt = os.listdir(self.path_gt)
        self.data_gt_depth = os.listdir(self.path_gt_depth)
        self.data_img_hist = os.listdir(self.path_img_hist)
        if mode == 1:
            self.ImgTrans = ImageTransform(loadSize)["train"]
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]

    def __len__(self):
        return len(self.data_gt)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path_img, self.data_img[idx])).convert("RGB")
        gt = Image.open(os.path.join(self.path_gt, self.data_img[idx])).convert("RGB")
        label_depth = Image.open(
            os.path.join(self.path_gt_depth, self.data_img[idx])
        ).convert("RGB")
        img_hist = Image.open(
            os.path.join(self.path_img_hist, self.data_img[idx])
        ).convert("RGB")

        name = self.data_img[idx]
        h, w = img.size

        if self.mode == 1:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.ImgTrans(img)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans(gt)
            torch.random.manual_seed(seed)
            label_depth = self.ImgTrans(label_depth)
            torch.random.manual_seed(seed)
            img_hist = self.ImgTrans(img_hist)

        else:
            img = self.ImgTrans(img)
            gt = self.ImgTrans(gt)
            label_depth = self.ImgTrans(label_depth)
            img_hist = self.ImgTrans(img_hist)

        return img, gt, label_depth, img_hist, name, (h, w)
