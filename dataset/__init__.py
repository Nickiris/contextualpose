
import torch

from .dataset import PoseDataset
from . import transforms as T
from .target_generator import HeatmapGenerator


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}


def make_train_dataloader(cfg, distributed=True):
    batch_size = cfg.BATCH_SIZE
    shuffle = True

    transforms = build_transforms(cfg)
    target_generator = HeatmapGenerator(cfg.DATASET.OUTPUT_SIZE)
    dataset = PoseDataset(cfg, is_train=True, transform=transforms, target_generator=target_generator)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler,
        collate_fn=trivial_batch_collator
    )

    return data_loader

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    images = []
    instances = []
    for (image, inst) in batch:
        images.append(image.unsqueeze(0))
        instances.append(inst)
    return images, instances

def make_test_dataloader(cfg):

    dataset = PoseDataset(cfg, is_train=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, data_loader

def build_transforms(cfg):

    max_rotation = cfg.DATASET.MAX_ROTATION
    min_scale = cfg.DATASET.MIN_SCALE
    max_scale = cfg.DATASET.MAX_SCALE
    max_translate = cfg.DATASET.MAX_TRANSLATE
    input_size = cfg.DATASET.INPUT_SIZE
    output_size = cfg.DATASET.OUTPUT_SIZE
    flip = cfg.DATASET.FLIP
    scale_type = cfg.DATASET.SCALE_TYPE

    if 'coco' in cfg.DATASET.DATASET:
        dataset_name = 'COCO'
    elif 'crowdpose' in cfg.DATASET.DATASET:
        dataset_name = 'CROWDPOSE'
    else:
        raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
    # if cfg.DATASET.WITH_CENTER:
    flip_index = FLIP_CONFIG[dataset_name]


    transforms = T.Compose(
        [
            T.RandomAffineTransform(
                input_size,
                output_size,
                max_rotation,
                min_scale,
                max_scale,
                scale_type,
                max_translate
            ),
            T.RandomHorizontalFlip(flip_index, output_size, flip),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return transforms
