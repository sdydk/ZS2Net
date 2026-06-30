import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import trange
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =True

class COCO_Dataset(Dataset):
    """
    NOTES：if you data style is coco, the NUM_CLASSES and CAT_LIST should be changes
    for example: the bacteria have 4 class (include background), the labels are [0, 1, 2, 3]
    """
    NUM_CLASSES = 39
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]

    def __init__(self, args, base_dir=r'./Datasets/ZMI2K/COCO2017', split='train', year='2017'):
        super().__init__()
        if split == 'test':
            ann_file = os.path.join(base_dir, 'annotations/image_info_{}-dev{}.json'.format(split, year))
            ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
            self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        else:
            ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
            ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
            self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))

        self.split = split
        # self.void_classes = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __getitem__(self, index):
        _img, _target, _img_serial_number = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'serial_number': _img_serial_number}
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)
        elif self.split == "test":
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img_serial_number = path.split('\\')[1].split(".")[0]
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width']))
        return _img, _target, img_serial_number

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])

            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.format(i, len(ids), len(new_ids)))
        print('Found number of qualified images:', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        compsed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), split='val'),
            tr.ToTensor(split='val')
        ])
        return compsed_transforms(sample)

    def __len__(self):
        return len(self.ids)