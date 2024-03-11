import logging
import time
import torch


class Trainer(object):
    def __init__(self, cfg, model, output_dir, writer_dict):
        self.cfg = cfg
        self.model = model
        self.output_dir = output_dir
        self.writer_dict = writer_dict
        self.weight_decay_coefficient = cfg.TRAIN.WD
        self.total_epoch = cfg.TRAIN.END_EPOCH

    def train(self, epoch, data_loader, optimizer, scheduler=None):
        logger = logging.getLogger("Training")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        multi_loss_meter = AverageMeter()
        single_loss_meter = AverageMeter()
        contrastive_loss_meter = AverageMeter()

        self.model.train()

        end = time.time()
        iterations = 0
        for i, (images, batched_inputs) in enumerate(data_loader):
            data_time.update(time.time() - end)

            images = torch.cat(images, dim=0)
            images = images.cuda(non_blocking=True)


            loss_dict = self.model(images, batched_inputs)

            loss = 0
            num_images = len(images)
            if 'multi_heatmap_loss' in loss_dict:
                multi_loss = loss_dict['multi_heatmap_loss']
                multi_loss_meter.update(multi_loss.item(), num_images)
                loss += multi_loss

            if 'single_loss' in loss_dict:
                single_loss = loss_dict['single_loss']
                single_loss_meter.update(single_loss.item(), num_images)
                loss += single_loss

            if 'contrastive_loss' in loss_dict:
                contrastive_loss = loss_dict['contrastive_loss']
                contrastive_loss_meter.update(contrastive_loss.item(), num_images)
                loss += contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step_iter()
            iterations += 1

            batch_time.update(time.time() - end)
            end = time.time()

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{multi}{contrastive}{single}'.format(
                epoch, i, len(data_loader),
                batch_time=batch_time,
                speed=num_images / batch_time.val,
                data_time=data_time,
                multi=_get_loss_info(multi_loss_meter, 'multi'),
                contrastive=_get_loss_info(contrastive_loss_meter, 'contrastive'),
                single=_get_loss_info(single_loss_meter, 'single'),
            )
            logger.info(msg)

        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['train_global_steps']
        writer.add_scalar('multi_loss', contrastive_loss_meter.val, global_steps)
        writer.add_scalar('contrastive_loss', contrastive_loss_meter.val, global_steps)
        writer.add_scalar('single_loss', single_loss_meter.val, global_steps)

        self.writer_dict['train_global_steps'] = global_steps + 1


def _get_loss_info(meter, loss_name):
    msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=loss_name, meter=meter)
    return msg


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



