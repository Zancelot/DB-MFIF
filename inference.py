import argparse
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torchvision.transforms as transforms
from imageio import imread
import numpy as np
from PIL import Image
import models


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, array):
        assert isinstance(array, np.ndarray)
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).float()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DBMFIFNet inference on a folder of img pairs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', metavar='DIR', default='/media/DATA3/zjc/LytroDataset/', help='path to images folder')
    parser.add_argument('--pretrained', metavar='PTH', default='/media/DATA3/zjc/DB_MFIF/checkpoint.pth.tar', help='path to pre-trained model')
    parser.add_argument('--output', metavar='DIR', default='/media/DATA3/zjc/DB_MFIF/Fusionnet_no138/', help='path to output folder')
    parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str,
                        help="images extensions to glob")
    return parser.parse_args()


def load_model(device, pretrained):
    network_data = torch.load(pretrained)
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True
    return model


def process_images(data_dir, model, device, save_path, img_exts):
    input_transform = transforms.Compose([
        ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    # TODO:
    # Modify the following line of code as needed to enable testing with different data sets.
    img_pairs = [(data_dir / f'lytro-{i:02d}-A.png', data_dir / f'lytro-{i:02d}-B.png') for i in range(1, 21)]

    for img1_file, img2_file in tqdm(img_pairs):
        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.cat([img1, img2]).unsqueeze(0).to(device)

        output = model(input_var)
        save_results(output, img1_file, save_path)


def save_results(output, img1_file, save_path):
    for fusion_output in output:
        results = fusion_output.detach().cpu().numpy()
        save_image(results[1:4, :], os.path.basename(img1_file).split('.')[0], save_path, 'DBMFIF')


def save_image(img_tensor, filename, save_path, suffix):
    filename = filename[:-1]
    to_save = np.clip(np.rint(img_tensor * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(to_save).save(save_path / f'{filename}{suffix}.png')


def main():
    args = parse_args()
    data_dir = Path(args.data)
    save_path = Path(args.output)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, args.pretrained)

    process_images(data_dir, model, device, save_path, args.img_exts)


if __name__ == '__main__':
    main()
