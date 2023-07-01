import logging
import os
import os.path
from random import choice
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PoseDataset(Dataset):

    def __init__(self, cfg, is_train, transform=None, target_generator=None):
        super(PoseDataset, self).__init__()
        self.root = cfg.DATASET.ROOT
        self.dataset = cfg.DATASET.DATASET
        if self.dataset == 'crowdpose':
            from crowdposetools.coco import COCO
        else:
            from pycocotools.coco import COCO
        self.image_set = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
        self.is_train = is_train
        self.transform = transform
        if 'test' in self.image_set:
            self.coco = COCO(os.path.join(self.root, 'annotations', 'image_info_test-dev2017.json'))
        else:
            self.coco = COCO(os.path.join(self.root, 'annotations', 'person_keypoints_{}.json'.format(self.image_set)))
        self.ids = list(self.coco.imgs.keys())

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        if is_train:
            if self.dataset == 'coco':
                self.filter_for_annotations()
            else:
                self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]
            self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS
            self.num_center = cfg.DATASET.NUM_CENTER
            if self.num_center == 3:
                self.center_idx = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16]]
            elif self.num_center == 5:
                self.center_idx = [[0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16]]

            self.output_size = cfg.DATASET.OUTPUT_SIZE
            self.heatmap_generator = target_generator

    def _get_image_path(self, file_name):
        images_dir = self.root
        if self.dataset == 'coco': images_dir = os.path.join(images_dir, '{}'.format(self.image_set))
        return os.path.join(images_dir, file_name)

    def filter_for_annotations(self, min_kp_anns=1):
        print('filter for annotations (min kp=%d) ...', min_kp_anns)

        def filter_image(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            anns = [ann for ann in anns if not ann.get('iscrowd')]
            if not anns:
                return False
            kp_anns = [ann for ann in anns
                       if 'keypoints' in ann and any(v > 0.0 for v in ann['keypoints'][2::3])]
            return len(kp_anns) >= min_kp_anns

        self.ids = [image_id for image_id in self.ids if filter_image(image_id)]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        file_name = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(
            self._get_image_path(file_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = {}
        if self.is_train:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            anno = [obj for obj in target]
            img_info = self.coco.loadImgs(img_id)[0]
            # mask
            m = np.zeros((img_info['height'], img_info['width']))
            if self.dataset == 'coco':
                import pycocotools
                for obj in anno:
                    if obj['iscrowd']:
                        rle = pycocotools.mask.frPyObjects(
                            obj['segmentation'], img_info['height'], img_info['width'])
                        m += pycocotools.mask.decode(rle)
                    elif obj['num_keypoints'] == 0:
                        rles = pycocotools.mask.frPyObjects(
                            obj['segmentation'], img_info['height'], img_info['width'])
                        for rle in rles:
                            m += pycocotools.mask.decode(rle)
            mask = m < 0.5

            anno = [obj for obj in anno if obj['iscrowd'] == 0 and obj['num_keypoints'] > 0]
            num_people = len(anno)
            area = np.zeros((num_people, 1))
            bboxs = np.zeros((num_people, 4, 2))
            keypoints = np.zeros((num_people, self.num_keypoints, 3))
            centers = np.zeros((num_people, self.num_center, 3))  # three centers

            for i, obj in enumerate(anno):
                keypoints[i, :, :3] = np.array(obj['keypoints']).reshape([-1, 3])
                area[i, 0] = obj['bbox'][2] * obj['bbox'][3]
                bboxs[i, :, 0], bboxs[i, :, 1] = obj['bbox'][0], obj['bbox'][1]
                bboxs[i, 1, 0] += obj['bbox'][2]
                bboxs[i, 2, 1] += obj['bbox'][3]
                bboxs[i, 3, 0] += obj['bbox'][2]
                bboxs[i, 3, 1] += obj['bbox'][3]

            if self.transform:
                img, mask, keypoints, area, bboxs = self.transform(img, mask, keypoints, area, bboxs)

            for i, obj in enumerate(anno):
                if not self.dataset == 'crowdpose':
                    if area[i, 0] < 32 ** 2:
                        centers[i, :, 2] = 0
                        continue
                vis = (keypoints[i, :, 2:3] > 0).astype(np.float32)
                if self.num_center == 1:
                    keypoints_sum = np.sum(keypoints[i, :, :2] * vis, axis=0)
                    num_vis_keypoints = len(np.nonzero(keypoints[i, :, 2])[0])
                    if num_vis_keypoints <= 0: centers[i, 0, 2] = 0; continue
                    centers[i, 0, :2] = keypoints_sum / num_vis_keypoints
                    centers[i, 0, 2] = 2
                    continue
                for j in range(self.num_center):
                    c = np.sum(keypoints[i, self.center_idx[j], :2] * vis[self.center_idx[j]], axis=0)
                    n = len(np.nonzero(keypoints[i, self.center_idx[j], 2])[0])
                    if n <= 0: centers[i, j, 2] = 0; continue
                    centers[i, j, :2] = c / n
                    centers[i, j, 2] = 2

            keypoints_with_centers = np.concatenate((keypoints, centers), axis=1)
            heatmap_with_centers, _ = self.heatmap_generator(keypoints_with_centers, bboxs)

            inst_coords, inst_heatmaps, inst_masks = self.get_inst_annos(centers, keypoints, area, bboxs)
            if len(inst_coords) > 0:
                inst_coords = np.concatenate(inst_coords, axis=0)
                inst_heatmaps = np.concatenate(inst_heatmaps, axis=0)
                inst_masks = np.concatenate(inst_masks, axis=0)
                results['instance_coord'] = torch.from_numpy(inst_coords)
                results['instance_heatmap'] = torch.from_numpy(inst_heatmaps)
                results['instance_mask'] = torch.from_numpy(inst_masks)
            results['multi_heatmap'] = torch.from_numpy(heatmap_with_centers)
            results['multi_mask'] = torch.from_numpy(mask[None, :, :])
        else:
            img = torch.from_numpy(img)
            results['image_id'] = img_id

        return img, results

    def __len__(self):
        return len(self.ids)

    def get_inst_annos(self, centers, keypoints, area, bbox):
        ind_vis = []
        area_idx = np.argsort(area.squeeze())
        inst_coords, inst_heatmaps, inst_masks = [], [], []
        for i in area_idx:
            inst_coord = []
            for j in range(self.num_center):
                center = centers[i, j]
                if center[2] < 1: continue
                x, y = int(center[0]), int(center[1])
                if x < 0 or x >= self.output_size or y < 0 or y >= self.output_size: continue
                # rand center point in 3x3 grid
                new_x = x + choice([-1, 0, 1])
                new_y = y + choice([-1, 0, 1])
                if new_x < 0 or new_x >= self.output_size or new_y < 0 or new_y >= self.output_size:
                    new_x = x
                    new_y = y
                x, y = new_x, new_y

                if [y, x] in ind_vis: continue
                inst_coord.append([y, x])
                ind_vis.append([y, x])
            inst_size = len(inst_coord)
            if inst_size > 0:
                inst_coords.append(np.array(inst_coord))
                inst_heatmap, inst_mask = self.heatmap_generator(keypoints[i:i + 1, :, :], bbox[i:i + 1, :, :])
                inst_heatmaps.append(np.tile(inst_heatmap[None, :, :, :], (inst_size, 1, 1, 1)))
                inst_masks.append(np.tile(inst_mask[None, :, :, :], (inst_size, 1, 1, 1)))

        return inst_coords, inst_heatmaps, inst_masks

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str
