import argparse
import os
import copy
import json
from collections import OrderedDict

import torch
import torch.optim
import torch.utils.data
import torchvision
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader
from core.evaluator import Evaluator
from dataset import make_test_dataloader
from utils.transforms import get_multi_scale_size, resize_align_multi_scale, get_final_preds
from utils.nms import oks_nms
import cv2




# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|---" * (num_values + 1) + "|")

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + "..."
    logger.info(
        "| "
        + full_arch_name
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )


def Validator(cfg, model, logger, final_output_dir, writer_dict=None):


    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)

    model.eval()

    _, data_loader = make_test_dataloader(cfg)
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
    )])

    all_preds = []

    data_loader = tqdm(data_loader, dynamic_ncols=True, ncols=70)
    for i, (image, batch_input) in enumerate(data_loader):
        img_id = batch_input['image_id'].item()
        image = image[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        with torch.no_grad():
            image_resized, center, scale = resize_align_multi_scale(image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0)
            image_resized = transforms(image_resized)

            inputs = image_resized.unsqueeze(0)
            if cfg.TEST.FLIP_TEST:
                flip_image = torch.flip(image_resized, [2]).unsqueeze(0)
                inputs = torch.cat([inputs, flip_image], dim=0)
            inputs = inputs.cuda()
            instances = model(inputs)

            if 'poses' not in instances: continue

            poses = instances['poses'].cpu().numpy()
            scores = instances['scores'].cpu().numpy()

            poses = get_final_preds(poses, center, scale, [base_size[0], base_size[1]])
            # perform nms
            keep, _ = oks_nms(poses, scores, cfg.TEST.OKS_SCORE, np.array(cfg.TEST.OKS_SIGMAS) / 10.0)

            for _keep in keep:
                all_preds.append({
                    "keypoints": poses[_keep][:, :3].reshape(-1, ).astype(float).tolist(),
                    "image_id": img_id,
                    "score": float(scores[_keep]),
                    "category_id": 1
                })

    if cfg.DATASET.TEST == 'val2017':
        evaluator = Evaluator(cfg, final_output_dir)
        info_str = evaluator.evaluate(all_preds)
        name_values = OrderedDict(info_str)
        perf_indicator = name_values['AP']

        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(logger, name_value, cfg.MODEL.NAME)
        else:
            _print_name_value(logger, name_values, cfg.MODEL.NAME)

        return perf_indicator



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
