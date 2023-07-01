from yacs.config import CfgNode

def get_cfg() -> CfgNode:
    from .default import _C

    return _C.clone()

def update_config(cfg, args):
    cfg.MODEL.NAME = args.model
    cfg.MODEL.PRETRAINED = args.pretrained
    cfg.DATASET.ROOT = args.root
    cfg.GPUS = args.gpus
    return cfg
