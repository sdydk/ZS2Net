import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=39)
    parser.add_argument('--model', default='ZPSSNets', help='create model name')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_file', type=str, default='./log/zooplankton/')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='Single', help='select multi loss function or signal loss')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--base_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=240703)
    parser.add_argument('--state', type=str, default='val')
    parser.add_argument('--weights', type=str, default=r'./checkpoint/hrnet_cs_8090_torch11.pth')

    parser.add_argument('--embed_dims', type=list, default=[32, 64, 160, 256])
    parser.add_argument('--depths', type=list, default=[3, 3, 5, 2])
    parser.add_argument('--mlp_ratios', type=list, default=[8, 8, 4, 4])
    parser.add_argument('--drop_rate', type=float, default=0.)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    parser.add_argument('--data_path', type=str, default=r'./Datasets/ZMI2K/COCO2017')
    parser.add_argument('--out_dir', type=str, default=r'./output')
    parser.add_argument('--save_dir', type=str, default=r'./output')
    parser.add_argument('--freeze-layer', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0, 1 or cpu)')
    parser.add_argument('--dataset', type=str, default='zooplankton', help='Datasets')  # Kvasir  zooplankton

    # Vision Transformer parameter
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--in_channel', type=int, default=3)
    args = parser.parse_args()
    return args