import argparse
import os
import pprint
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import models
from config import get_cfg, update_config
from core.valid import Validator
from utils.logger import create_checkpoint, setup_logger


def parse_args():

    parser = argparse.ArgumentParser(description='Val')
    # general
    parser.add_argument('--val_path',
                        help='path of val data',
                        type=str)
    parser.add_argument('--model_path',
                        help='path of trained model',
                        type=str)
    args = parser.parse_args()
    return args


def main_worker(final_output_dir):
    args = parse_args()
    cfg = get_cfg()
    cfg.DATASET.TEST = args.val_path
    cfg.TEST.MODEL_FILE  = args.model_path
    # setup logger
    logger, _ = setup_logger(final_output_dir, cfg.RANK, 'val')
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    logger.info("torch backbends....")

    print("Use GPU: {} for training".format(cfg.RANK))

    model = models.create(cfg.MODEL.NAME, cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'log')),
        'train_global_steps': 0,
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info("device is: {}....".format(device))
    model = model.to(device)

    Validator(cfg, model, logger, final_output_dir)
    writer_dict['writer'].close()


if __name__ == '__main__':
    final_output_dir = "experiment"
    main_worker(final_output_dir)