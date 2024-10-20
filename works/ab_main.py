import logging
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import random
import time

import base
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load import OH, OW, TMAP, TestDataSet, TrainDataSet, ValDataSet
from parsearg import args as TRAIN_ARGS
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from codes.config import get_config
from codes.dataloaders.dataset import RandomGenerator, TwoStreamBatchSampler
from codes.networks.net_factory import net_factory
from codes.networks.vision_transformer import SwinUnet as ViT_seg
from codes.utils import losses, ramps
from codes.val_2D import test_single_volume


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


_args = {
    "patch_size": [OW, OH],
    "exp": "miccai",
    "model": "unet",
    "max_iterations": 3000000,
    "batch_size": 16,
    "deterministic": 1,
    "base_lr": 0.01,
    "seed": 1001,
    "num_classes": len(TMAP["to_cls"]) + 1,
    "cfg": "../codes/configs/swin_tiny_patch4_window7_224_lite.yaml",
    "opts": None,
    "labeled_bs": 16,
    "labeled_num": 30,
    "ema_decay": 0.99,
    "consistency_type": "mse",
    "consistency": 0.1,
    "consistency_rampup": 200.0,
    "zip": False,
}
for k, v in _args.items():
    setattr(TRAIN_ARGS, k, v)
args = TRAIN_ARGS
config = get_config(args)
config.defrost()
config.DATA.IMG_SIZE = OW
config.freeze()


def create_model(num_classes, ema=False):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model1 = create_model(args.num_classes)
    model2 = ViT_seg(
        config, img_size=args.patch_size, num_classes=args.num_classes
    ).cuda()
    model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = TrainDataSet(
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = ValDataSet(db_train)

    total_slices = len(db_train)
    labeled_slice = args.labeled_num
    print(
        "Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice)
    )
    labeled_idxs = list(range(0, labeled_slice))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, labeled_idxs, batch_size, batch_size // 2
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(
        model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    optimizer2 = optim.SGD(
        model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            loss1 = 0.5 * (
                ce_loss(
                    outputs1[: args.labeled_bs], label_batch[: args.labeled_bs].long()
                )
                + dice_loss(
                    outputs_soft1[: args.labeled_bs],
                    label_batch[: args.labeled_bs].unsqueeze(1),
                )
            )
            loss2 = 0.5 * (
                ce_loss(
                    outputs2[: args.labeled_bs], label_batch[: args.labeled_bs].long()
                )
                + dice_loss(
                    outputs_soft2[: args.labeled_bs],
                    label_batch[: args.labeled_bs].unsqueeze(1),
                )
            )

            model1_loss = loss1
            model2_loss = loss2

            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group["lr"] = lr_
            for param_group in optimizer2.param_groups:
                param_group["lr"] = lr_

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/model1_loss", model1_loss, iter_num)
            writer.add_scalar("loss/model2_loss", model2_loss, iter_num)
            logging.info(
                "iteration %d : model1 loss : %f model2 loss : %f"
                % (iter_num, model1_loss.item(), model2_loss.item())
            )
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                outputs = torch.argmax(
                    torch.softmax(outputs1, dim=1), dim=1, keepdim=True
                )
                writer.add_image(
                    "train/model1_Prediction", outputs[1, ...] * 50, iter_num
                )
                outputs = torch.argmax(
                    torch.softmax(outputs2, dim=1), dim=1, keepdim=True
                )
                writer.add_image(
                    "train/model2_Prediction", outputs[1, ...] * 50, iter_num
                )
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model1,
                        classes=num_classes,
                        patch_size=args.patch_size,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model1_val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],  # type: ignore[index]
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/model1_val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],  # type: ignore[index]
                        iter_num,
                    )

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/model1_val_mean_dice", performance1, iter_num)
                writer.add_scalar("info/model1_val_mean_hd95", mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model1_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance1, 4)
                        ),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model1.pth".format(args.model)
                    )
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    "iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f"
                    % (iter_num, performance1, mean_hd951)
                )
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model2,
                        classes=num_classes,
                        patch_size=args.patch_size,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model2_val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],  # type: ignore[index]
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/model2_val_{}_hd95".format(class_i + 1),
                        metric_list[class_i, 1],  # type: ignore[index]
                        iter_num,
                    )

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/model2_val_mean_dice", performance2, iter_num)
                writer.add_scalar("info/model2_val_mean_hd95", mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model2_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance2, 4)
                        ),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model2.pth".format(args.model)
                    )
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    "iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f"
                    % (iter_num, performance2, mean_hd952)
                )
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "model1_iter_" + str(iter_num) + ".pth"
                )
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, "model2_iter_" + str(iter_num) + ".pth"
                )
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


train(args, "ab_snapshots")
