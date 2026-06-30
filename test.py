import numpy as np
import torch
import os
import argparse
from PIL import Image
from torch.utils import data
from torchvision import transforms
from model.FCN import FCNs
from model.UNet import UNet
from model.DeepLabv3Plus import DeepLabv3plus
from model.PSPNet import PSPNets
from model.FPN import FPNs
from model.OCRNet import OCRNets
from model.ZS2Net import ZPSSNets
from utils.coco import COCO_Dataset

from utils.utils import decode_segmap
from torch.utils.data import DataLoader
from utils.argument import parse_args
from metrics import iou, pixel_acc, boundary_iou
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
args = parse_args()
cityscapes_trainIds2labelIds = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
    model_name = "FCNs"
    dataset_name = "cityscapes"
    loss_name = "CEL"
    pretrained_if = "F"

    PATH = './output/{}_{}_{}_{}.pth'.format(model_name, dataset_name, loss_name, pretrained_if)
    N_CLASS = 19
    BATCH_SIZE = 8
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 224
    args.crop_size = 224
"""

model = None
if args.model == "FCNs":
    model = FCNs(
        args
    )
elif args.model == "UNet":
    model = UNet(
        n_class=args.num_class
    )
elif args.model == "DeepLabv3plus":
    model = DeepLabv3plus(
        n_class=args.num_class,
        in_channel=args.in_channel,
        pretrained=args.pretrained
    )
elif args.model == "PSPNets":
    model = PSPNets(
        in_channel=args.in_channel,
        pretrained=args.pretrained,
        name="ResNet101",
        out_channels=args.num_class
    )
elif args.model == "FPNs":
    model = FPNs(
        num_class=args.num_class,
        pretrained=args.pretrained
    )
elif args.model == "OCRNet":
    model = OCRNets(
        args
    )
elif args.model == "ZPSSNets":
    model = ZPSSNets(
        num_class=args.num_class,
        pretrained=args.pretrained
    )

train_ds = COCO_Dataset(args, base_dir=args.data_path, split='train', year='2017')
test_ds = COCO_Dataset(args, base_dir=args.data_path, split='{}'.format(args.state), year='2017')
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)


PATH = '{}/{}/{}/{}_{}_{}K_{}_{}_{}_{}.pth'.format(args.out_dir, args.dataset, args.model, args.model,
                                                   args.base_size, int((args.epochs / 1000) * len(train_dl)),
                                                   args.batch_size, args.loss, args.optimizer, args.pretrained)
print(PATH)
model.load_state_dict(torch.load(PATH), strict=False)
# model = model.to(device)

def plot_picture():
    # ./ output
    save_path = '{}/{}/{}/{}/{}'.format(args.save_dir, args.dataset, args.model, args.loss, args.state)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    total_ious = []
    pixel_accs = []
    boundary_ious = []
    model.eval()
    count = 1
    for _, sample in enumerate(test_dl):

        images_batch, mask_batch, images_original, serial_number = sample['image'], sample['label'], \
            sample['original'], sample['serial_number']
        if args.model == "OCRNet":
            pred_batch, _ = model(images_batch)
        else:
            pred_batch = model(images_batch)

        pred_batch = pred_batch.data.cpu().numpy()
        N, _, h, w = pred_batch.shape
        pred_batch = pred_batch.transpose(0, 2, 3, 1).reshape(-1, args.num_class).argmax(axis=1).reshape(N, h, w)
        target = mask_batch.cpu().numpy().reshape(N, h, w)

        figsize = (16, 12)
        for i in range(len(images_original)):
            plt.figure()
            # plt.title('Image', y=-0.2)
            plt.xticks([])
            plt.yticks([])
            # plt.axis('off')
            plt.imshow(images_original[i].permute(1, 2, 0).cpu().numpy().astype('uint8'))  # .astype('uint8')
            # plt.tight_layout()
            # plt.show()
            plt.savefig('{}/{}_1.jpg'.format(save_path, serial_number[i]), dpi=300, bbox_inches='tight', pad_inches=0)

            plt.figure()
            # plt.title('Ground Truth', y=-0.2)
            mask = decode_segmap(label_mask=target[i], dataset=args.dataset, plot=False)
            # mask = cv2.resize(mask, (images_original[i].shape[2], images_original[i].shape[1]), interpolation=cv2.INTER_NEAREST)
            plt.xticks([])
            plt.yticks([])
            # plt.axis('off')
            plt.imshow(mask.astype('uint8'))
            # plt.tight_layout()
            # plt.show()
            plt.savefig('{}/{}_2.jpg'.format(save_path, serial_number[i]), dpi=300, bbox_inches='tight', pad_inches=0)

            plt.figure()
            # plt.title('{}'.format(args.model), y=-0.2)
            pred_label = decode_segmap(label_mask=pred_batch[i], dataset=args.dataset, plot=False)
            plt.xticks([])
            plt.yticks([])
            # plt.axis('off')
            plt.imshow(pred_label)
            # plt.tight_layout()
            # plt.show()
            plt.savefig('{}/{}_3.jpg'.format(save_path, serial_number[i]), dpi=300, bbox_inches='tight', pad_inches=0)

            count += 1
            # plt.show()
    print('Done!')

def print_miou_pixacc():
    save_path = './output/{}/{}/{}/{}'.format(args.dataset, args.model, args.loss, args.state)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    total_ious = []
    pixel_accs = []
    boundary_ious = []
    model.eval()
    count = 1
    for _, sample in enumerate(test_dl):

        images_batch, mask_batch, images_original = sample['image'], sample['label'], sample['original']
        if args.model == "OCRNet":
            pred_batch, _ = model(images_batch)
        else:
            pred_batch = model(images_batch)

        pred_batch = pred_batch.data.cpu().numpy()
        N, _, h, w = pred_batch.shape
        pred_batch = pred_batch.transpose(0, 2, 3, 1).reshape(-1, args.num_class).argmax(axis=1).reshape(N, h, w)
        target = mask_batch.cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred_batch, target):
            total_ious.append(iou(p, t, args.num_class))
            pixel_accs.append(pixel_acc(p, t))
            boundary_ious.append(boundary_iou(gt=p, dt=t, dilation_ratio=0.005, cls_num=args.num_class))
    total_ious = np.array(total_ious).T
    mIoU = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    boundary_ious = np.array(boundary_ious).T
    bmIoU = np.nanmean(boundary_ious, axis=1)

    print('PA:{}, IoU: {}, bIoU:{}'.format(pixel_accs, np.nanmean(mIoU), np.nanmean(bmIoU)))
    print('Done!')

if __name__ == '__main__':
    # 绘图
    # plot_picture()
    # 评价指标：PA, mIoU, bIoU
    print_miou_pixacc()