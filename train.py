import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from thop import profile
from thop import clever_format
import pytorch_warmup as warmup
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model.FCN import FCNs
from model.UNet import UNet
from model.DeepLabv3Plus import DeepLabv3plus
from model.PSPNet import PSPNets
from model.FPN import FPNs
from model.OCRNet import OCRNets
from model.ZS2Net import ZS2Net
from torch.autograd import Variable
from utils.coco import COCO_Dataset

from utils.argument import parse_args
from utils.random_seed import setup_seed
from metrics import iou, pixel_acc
from torch.utils.data import DataLoader
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")
import wcwidth
table_header = ['name', 'IoU']
class_index = [
    '_background_', 'Calanus sinicus', 'Sagitta crassa', 'Themisto gracilipes', 'Penilia avirostris',
    'Centropages abdominalis', 'Acartia pacifica', 'Centropages tenuiremis', 'Pontellopsis tenuicauda',
    'Calanopia thompsoni', 'Sugiura chengshanense', 'Ophioplutues larva early', 'Eirene menoni',
    'Euphausia pacifica', 'Evadne tergestina', 'Muggiaea atlantica', 'Paracalanus parvus', 'Oithona plumifera',
    'Pleurobrachia globosa', 'Clytia folleata', 'Obelia dichotoma', 'Ectopleura bimanatus', 'Doliolum denticulatum',
    'Oikopleura longicauda', 'Tornaria larva', 'Polychaeta larva early', 'Polychaeta larva later',
    'Turritopsis nutricula', 'Proboscidactyla flavicirrata', 'Fritillaria formica', 'Labidocera rotunda',
    'Alima larva', 'Megalopa larva', 'Brachyura zoea larva', 'Ophioplutues larva later', 'Fish eggs', 'Fish larva',
    'Actinotrocha larva', 'Trochophora larva'
]

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
Console = logging.StreamHandler()
Console.setLevel(logging.INFO)
def train(args, model, loss_fn, optimizer, epochs, train_loader, val_loader, n_classes, PATH):
    writer = SummaryWriter('./run/{}/{}'.format(args.dataset, args.model))
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    running_loss, best_acc, train_acc, best_miou = 0.0, 0.0, 0.0, 0.0
    # 学习率衰减策略
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 学习率预热策略
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    logger.info("Start Train!!!")
    start_total_time = time.time()
    for epoch in range(epochs):
        start_time = time.time()
        train_bar = tqdm(train_loader)
        correct = 0
        total = 0
        running_loss = 0
        model.train()
        for _, sample in enumerate(train_bar):
            optimizer.zero_grad()
            x, y = sample['image'], sample['label']
            if torch.cuda.is_available():
                x, y = x.to('cuda'), y.to('cuda')
            if args.model == "OCRNet":
                y_predicted, probs = model(x)
            else:
                y_predicted = model(x)

            loss = loss_fn(y_predicted, y.long()).to('cuda')  #
            loss.backward()
            optimizer.step()

            with warmup_scheduler.dampening():
                exp_lr_scheduler.step()

            y_predicted = torch.argmax(y_predicted, dim=1)
            correct += (y_predicted == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()

            # train_acc += torch.eq(y_predicted, y).sum().item()
            train_bar.desc = "train epoch[{}/{}] | loss:{:.3f} ".format(epoch+1, epochs, loss)

        train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        train_acc = correct / (total * 256 * 256)

        # train_accurate = train_acc / len(train_loader)
        # acc_train.append(train_accurate)
        model.eval()
        # val_acc = 0.0
        val_correct = 0
        val_total = 0
        val_running_loss = 0
        total_ious = []
        pixel_accs = []

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for _, sample in enumerate(val_bar):
                x, y = sample['image'], sample['label']
                if torch.cuda.is_available():
                    x, y = x.to('cuda'), y.to('cuda')
                if args.model == "OCRNet":
                    y_predicted, probs = model(x)
                else:
                    y_predicted = model(x)
                loss = loss_fn(y_predicted, y.long()).to('cuda') #
                pred_batch = y_predicted.data.cpu().numpy()
                N, _, h, w = pred_batch.shape
                pred_batch = pred_batch.transpose(0, 2, 3, 1).reshape(-1, n_classes).argmax(axis=1).reshape(N, h, w)
                target = y.cpu().numpy().reshape(N, h, w)
                for p, t in zip(pred_batch, target):
                    total_ious.append(iou(p, t, n_classes))
                    pixel_accs.append(pixel_acc(p, t))
                y_predicted = torch.argmax(y_predicted, dim=1)
                val_correct += (y_predicted == y).sum().item()
                val_total += y.size(0)
                val_running_loss += loss.item()

        writer.add_scalars('loss', {'val': val_running_loss}, epoch)
        total_ious = np.array(total_ious).T
        mIoU = np.nanmean(total_ious, axis=1)
        pixel_accs = np.array(pixel_accs).mean()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / (val_total * 256 * 256)

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        end_time = time.time() - start_time
        writer.add_scalars('lr', {'lr': optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
        print('[epoch {:d}] | train_loss: {:.3f} | val_accuracy: {:.3f} | pix_acc: {:.3f} | mIoU: {} '
              '| epoch_time: {:.0f}m {:.0f}s | lr:{:0.6f}'.
              format(epoch + 1, train_loss, val_acc, pixel_accs, np.nanmean(mIoU),
                     (end_time // 60), (end_time % 60), optimizer.state_dict()['param_groups'][0]['lr']))
        print('IoU: {}'.format(mIoU))
        iou_single_class = np.vstack(np.array((class_index, mIoU))).T
        if pixel_accs > best_acc:
            if np.nanmean(mIoU) > best_miou:
                best_acc = pixel_accs
                best_miou = np.nanmean(mIoU)
                logger.info("[epoch {:d}] | pixel_accs: {:.3f},  mIoU: {}"
                            .format(epoch + 1, pixel_accs, np.nanmean(mIoU)))
                # print(tabulate(mIoU, headers=table_header, tablefmt='grid'))
                logger.info(tabulate(iou_single_class, headers=table_header, tablefmt='grid'))
                pth_file = '{}_{}_{}K_{}_{}_{}_{}.pth'.format(args.model, args.base_size,
                                                              int((args.epochs / 1000) * len(train_loader)),
                                                              args.batch_size, args.loss, args.optimizer,
                                                              args.pretrained)
                pth_path = os.path.abspath(os.path.join(PATH, pth_file))
                torch.save(model.state_dict(), pth_path)
        elif pixel_accs == best_acc:
            if np.nanmean(mIoU) > best_miou:
                best_acc = pixel_accs
                best_miou = np.nanmean(mIoU)
                logger.info("[epoch {:d}] | pixel_accs: {:.3f},  mIoU: {}"
                            .format(epoch + 1, pixel_accs, np.nanmean(mIoU)))
                # print(tabulate(mIoU, headers=table_header, tablefmt='grid'))
                logger.info(tabulate(iou_single_class, headers=table_header, tablefmt='grid'))
                pth_file = '{}_{}_{}K_{}_{}_{}_{}.pth'.format(args.model, args.base_size,
                                                              int((args.epochs / 1000) * len(train_loader)),
                                                              args.batch_size, args.loss, args.optimizer,
                                                              args.pretrained)
                pth_path = os.path.abspath(os.path.join(PATH, pth_file))
                torch.save(model.state_dict(), pth_path)

    writer.close()
    end_total_time = time.time() - start_total_time
    logger.info('Finished Training!!!, Total time: {:.0f}m {:.0f}s'.format(end_total_time // 60, end_total_time % 60))

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.data_path):
        print('Dataset file is not right!!!')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.seed is not None:
        setup_seed(args.seed)
    # / {}  ,
    log_file = os.path.join(args.log_file, "{}".format(args.model), "{}.txt".format(
        time.strftime('%Y-%m-%d_%H%M%S', time.localtime())
    ))

    dir_name = os.path.dirname(os.path.abspath(log_file))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    Handler = logging.FileHandler(log_file, mode='a')
    Handler.setLevel(logging.INFO)
    Handler.setFormatter(formatter)
    logger.addHandler(Console)
    logger.addHandler(Handler)

    model = None
    if args.model == "FCNs":
        model = FCNs(args)
    elif args.model == "UNet":
        model = UNet(n_class=args.num_class)
    elif args.model == "DeepLabv3plus":
        model = DeepLabv3plus(n_class=args.num_class, in_channel=args.in_channel, pretrained=args.pretrained)
    elif args.model == "PSPNets":
        model = PSPNets(in_channel=args.in_channel, pretrained=args.pretrained, name="ResNet101", out_channels=args.num_class)
    elif args.model == "FPNs":
        model = FPNs(num_class=args.num_class, pretrained=args.pretrained)
    elif args.model == "OCRNet":
        model = OCRNets(args)
    elif args.model == "ZS2Net":
        model = ZS2Net(num_class=args.num_class, pretrained=args.pretrained)

    logger.info("model name: {}".format(args.model))
    logger.info("Net Struction: {}".format(model))
    train_ds = COCO_Dataset(args, base_dir=args.data_path, split='train', year='2017')
    val_ds = COCO_Dataset(args, base_dir=args.data_path, split='val', year='2017')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("using {} images for training, {} images for validation.".format(len(train_ds), len(val_ds)))
    loss_fn = nn.CrossEntropyLoss()

    optimizer = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0001)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8, weight_decay=0.0005)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.1,
                                      amsgrad=False)
    optimizer.zero_grad()
    logger.info("epochs: {}".format(args.epochs))
    logger.info("iteration: {}K".format(int((args.epochs / 1000) * len(train_dl))))
    logger.info("batch_size: {}".format(args.batch_size))
    logger.info("img_size: {}".format(args.img_size))
    logger.info("loss function: {}".format(args.loss))
    logger.info("optimizer: {}".format(args.optimizer))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.to('cuda')

    input = torch.randn(1, 3, args.base_size, args.base_size).to(device)
    flops, params = profile(model, inputs=(input,))
    gflops, params = clever_format([flops * 2, params], "%.3f")
    logger.info("Model GFLOPs:{:} Params:{:}".format(gflops, params))

    train(args, model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=args.epochs,
          train_loader=train_dl, val_loader=val_dl, n_classes=args.num_class,
          PATH='{}/{}/{}'.format(args.out_dir, args.dataset, args.model))



