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
from core.trainer import Trainer
from dataset import make_train_dataloader
from utils.logger import create_checkpoint, setup_logger
from utils.utils import get_optimizer, save_checkpoint, LinearWarmupStepLR


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # general
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    # distributed training
    parser.add_argument('--gpus',
                        help='gpu ids for ddp training',
                        type=str)
    parser.add_argument('--model',
                        default='ContextualPose',
                        help='model for ddp training',
                        type=str)
    parser.add_argument('--pretrained',
                        help='pretrained model path',
                        type=str)
    parser.add_argument('--root',
                        help='root of dataset',
                        type=str)
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--port',
                        default='23459',
                        type=str,
                        help='port used to set up distributed training')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = get_cfg()
    cfg = update_config(cfg, args)

    final_output_dir = create_checkpoint(cfg, 'train')

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    dist_url = args.dist_url + ':{}'.format(args.port)

    ngpus_per_node = torch.cuda.device_count()
    if cfg.DDP:
        world_size = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(world_size, dist_url, final_output_dir, args))
    else:
        main_worker(0, 1, dist_url, final_output_dir, args)


def main_worker(rank, world_size, dist_url, final_output_dir, args):
    cfg = get_cfg()

    # setup logger
    logger, _ = setup_logger(final_output_dir, rank, 'train')
    if not cfg.DDP or (cfg.DDP and rank == 0):
        logger.info(pprint.pformat(args))
        logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    print("Use GPU: {} for training".format(rank))
    if cfg.DDP:
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.format(dist_url, world_size, rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=dist_url,
            world_size=world_size,
            rank=rank
        )

    model = models.create(cfg.MODEL.NAME, cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=os.path.join(final_output_dir, 'tblog')),
        'train_global_steps': 0
    }

    if cfg.DDP:
        if cfg.MODEL.SYNC_BN:
            print('use sync bn')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    train_loader = make_train_dataloader(cfg)
    logger.info(train_loader.dataset)

    last_epoch = -1
    optimizer = get_optimizer(cfg, model.parameters())

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'model', 'checkpoint.pth.tar')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    #     lr_scheduler = LinearWarmupStepLR(
    #             optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
    #             last_epoch=last_epoch
    #         )

    trainer = Trainer(cfg, model, final_output_dir, writer_dict)
    best_model = True

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        logger.info('=> learning rate is {}.'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        if cfg.DDP:
            train_loader.sampler.set_epoch(epoch)
        trainer.train(epoch, train_loader, optimizer, lr_scheduler)

        lr_scheduler.step()

        if not cfg.DDP or (cfg.DDP and rank == 0):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(rank)
    )
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()