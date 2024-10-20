import json
import os
from pathlib import Path
from random import shuffle
from typing import Any, Union

import numpy as np
import torch
from matplotlib.path import Path as GeoPath
from PIL import Image
from torch.utils.data import Dataset

OW = 224
OH = 224


with open(Path(__file__).parent.joinpath("tmap.json"), encoding="utf-8") as fp:
    TMAP = json.load(fp)


def to_teeth_cls(teeth_tag: Union[str, int]) -> int:
    return TMAP["to_cls"][str(teeth_tag)]


def to_teeth_tag(teeth_cls: Union[str, int]) -> int:
    return TMAP["to_tag"][str(teeth_cls)]


def get_image(p: Path) -> np.ndarray:
    img = Image.open(p).convert("L").resize((OW, OH))
    img = np.array(img, dtype=np.float32) / 255
    return img


def get_label(p: Path) -> tuple[str, np.ndarray]:
    with open(p, "r", encoding="utf-8") as fp:
        obj = json.load(fp)
    img_path = obj["imagePath"]
    w, h = obj["imageWidth"], obj["imageHeight"]
    label = np.zeros((OW, OH), dtype=np.uint8)

    _x, _y = np.meshgrid(np.arange(OW), np.arange(OH))
    _x, _y = _x.flatten(), _y.flatten()
    _points = np.vstack((_x, _y)).T

    for shape in obj["shapes"]:
        assert shape["shape_type"] == "polygon"
        polygon = tuple((x / w * OW, y / h * OH) for x, y in shape["points"])
        grid = GeoPath(polygon).contains_points(_points)
        mask = grid.reshape(OW, OH)

        _cls = to_teeth_cls(shape["label"])
        label[mask] = _cls

    return img_path, label


def get_empty_label() -> np.ndarray:
    return np.zeros((OW, OH), dtype=np.uint8)


class TrainDataSet(Dataset):
    def __init__(self, transform: Any = None) -> None:
        super().__init__()
        self.transform = transform

        self.ul_dir = Path(__file__).parent.parent.joinpath("mydata/Train-Unlabeled")
        self.l_dir = Path(__file__).parent.parent.joinpath("mydata/Train-Labeled")
        self.l_img_dir = self.l_dir.joinpath("Images")
        self.l_mask_dir = self.l_dir.joinpath("Masks")
        self.ul_img_dir = self.ul_dir

        self.l_datas: list[dict[str, np.ndarray]]
        if Path("labeled.npy").exists():
            self.l_datas = np.load("labeled.npy", allow_pickle=True)
        else:
            self.l_datas = []
            for p in os.listdir(self.l_mask_dir):
                img_name, label = get_label(self.l_mask_dir.joinpath(p))
                img = get_image(self.l_img_dir.joinpath(img_name))
                self.l_datas.append({"image": img, "label": label})
            np.save("labeled.npy", self.l_datas)

        self.ul_datas: list[dict[str, np.ndarray]]
        if Path("unlabeled.npy").exists():
            self.ul_datas = np.load("unlabeled.npy", allow_pickle=True)
        else:
            self.ul_datas = []
            for p in os.listdir(self.ul_img_dir):
                img = get_image(self.ul_img_dir.joinpath(p))
                self.ul_datas.append({"image": img, "label": get_empty_label()})
            np.save("unlabeled.npy", self.ul_datas)

        self.datas = np.concatenate((self.l_datas, self.ul_datas), axis=0)
        print("已完成数据集初始化")

    def __len__(self) -> int:
        return len(self.l_datas) + len(self.ul_datas)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.datas[idx]
        sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class ValDataSet(Dataset):
    def __init__(self, train_set: TrainDataSet) -> None:
        super().__init__()
        self.train_set = train_set

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_batch = tuple(
            torch.from_numpy(sample["image"]).unsqueeze(0)
            for sample in self.train_set.l_datas
        )
        label_batch = tuple(
            torch.from_numpy(sample["label"]).unsqueeze(0)
            for sample in self.train_set.l_datas
        )
        image = torch.concat(img_batch, dim=0).numpy()
        label = torch.concat(label_batch, dim=0).numpy()
        sample = {"image": image, "label": label}
        sample["idx"] = idx  # type: ignore
        return sample


class TestDataSet(Dataset):
    def __init__(self, train_set: TrainDataSet) -> None:
        super().__init__()
        self.train_set = train_set

    def __len__(self) -> int:
        return len(self.train_set.ul_datas)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.train_set.ul_datas[idx]
        image, label = sample["image"], sample["label"]
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


if __name__ == "__main__":
    TrainDataSet()
