import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torchinfo import summary

import numpy as np
from PIL import Image

sys.path.append("..")
sys.path.insert(1, "../codes")

import torch
from torch.utils.data import DataLoader, Dataset

from codes.networks.net_factory import net_factory

with open(Path(__file__).parent.joinpath("tmap.json"), encoding="utf-8") as fp:
    TMAP = json.load(fp)


@dataclass
class PolygonPoint:
    x: float
    y: float


def get_polygon_points(
    logits: torch.Tensor, ox: int, oy: int
) -> dict[int, list[PolygonPoint]]:
    points: dict[int, list[PolygonPoint]] = {}
    logitsmap = logits.squeeze().transpose(0, 1)

    pm = torch.nn.functional.pad(logitsmap, (1, 1, 1, 1), mode="constant", value=0)
    top = pm[:-2, 1:-1]
    left = pm[1:-1, :-2]
    right = pm[1:-1, 2:]
    bottom = pm[2:, 1:-1]
    cen = pm[1:-1, 1:-1]
    res: torch.Tensor = (
        (cen != top) | (cen != left) | (cen != right) | (cen != bottom)
    ) & (cen != 0)

    resized_logits = torch.from_numpy(
        np.array(
            Image.fromarray(logitsmap.cpu().numpy().astype("uint8")).resize(
                (logitsmap.shape[-1], logitsmap.shape[-2] * (ox // oy)),
                resample=Image.NEAREST,
            )
        )
    )
    imgmask = torch.from_numpy(
        np.array(
            Image.fromarray(res.cpu().numpy().astype("uint8")).resize(
                (logitsmap.shape[-1], logitsmap.shape[-2] * (ox // oy)),
                resample=Image.NEAREST,
            )
        )
    )
    pm = torch.nn.functional.pad(imgmask, (1, 1, 1, 1), mode="constant", value=0)
    top = pm[:-2, 1:-1]
    left = pm[1:-1, :-2]
    right = pm[1:-1, 2:]
    bottom = pm[2:, 1:-1]
    cen = pm[1:-1, 1:-1]
    imgmask = ((cen != top) | (cen != left) | (cen != right) | (cen != bottom)) & (
        cen != 0
    )

    for rx, ry in torch.nonzero(imgmask, as_tuple=False):
        x = round(float(rx * (ox / resized_logits.shape[-2])), 4)
        y = round(float(ry * (oy / resized_logits.shape[-1])), 4)
        try:
            teeth_tag = TMAP["to_tag"][str(int(resized_logits[rx, ry]))]
        except KeyError:
            continue
        points.setdefault(teeth_tag, []).append(PolygonPoint(x, y))

    return points


class SimpleDataset(Dataset):
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.files = tuple(
            self.folder.joinpath(filename) for filename in os.listdir(self.folder)
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, idx: int
    ) -> tuple[Path, torch.Tensor, torch.Tensor, torch.Tensor]:
        img = Image.open(self.files[idx])
        ox, oy = img.size
        img = img.convert("L").resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255
        torch_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return (
            torch.Tensor([idx]),
            torch.Tensor([ox]),
            torch.Tensor([oy]),
            torch_img,
        )


def export_json(
    out_folder: Path,
    dataset: SimpleDataset,
    points: dict[int, list[PolygonPoint]],
    data_idx: int,
    x: int,
    y: int,
) -> None:
    output: dict[str, Any] = {"imageHeight": y, "imageWidth": x}
    shapes: list[dict] = []
    for tag, ps in points.items():
        shapes.append({"label": str(tag), "points": [[p.x, p.y] for p in ps]})
    output["shapes"] = shapes

    basename = dataset.files[data_idx].parts[-1].split(".")[0]
    name_li = basename.split("_")
    filename = "_".join((name_li[-2], f"{int(name_li[-1]):04d}", "Mask.json"))
    with open(out_folder.joinpath(filename), "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)
    print(filename)


def get_output(img_folder: Path, out_folder: Path) -> list[PolygonPoint]:
    dataset = SimpleDataset(img_folder)
    dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=16,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(1001 + worker_id),
    )

    num_classes = len(TMAP["to_cls"]) + 1
    net = net_factory(net_type="unet", in_chns=1, class_num=num_classes)
    net.load_state_dict(
        torch.load(
            "/home/melodyecho/miccai/SSL4MIS/works/snapshots/model1_iter_100200_dice_0.9905.pth"
        )
    )
    net.eval()

    for idxes, xs, ys, batch in dataloader:
        print(batch.shape)
        summary(net, input_size=(16, 1, 224, 224))
        exit(0)
        batch = batch.cuda()
        output = net(batch)
        logits = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)

        for i in range(len(batch)):
            points = get_polygon_points(logits[i], int(xs[i]), int(ys[i]))
            export_json(
                out_folder, dataset, points, int(idxes[i]), int(xs[i]), int(ys[i])
            )

    return []


get_output(
    Path("/home/melodyecho/miccai/SSL4MIS/mydata/Validation-Public"),
    Path("/home/melodyecho/miccai/SSL4MIS/output"),
)
