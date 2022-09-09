import argparse
import imageio
from os import listdir
from os.path import join
from torchvision import transforms
import numpy as np
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str, required=True,
                        help='path to images to convert')
    parser.add_argument('--saving_path', type=str, required=True,
                        help='path to save new images')
    parser.add_argument('--resize', type=list, default=[-1],
                        help='resize images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    img_base_path = args.imgs_path
    saving_path = args.saving_path
    imgs_filenames = [str(l) for l in listdir(img_base_path) if str(l).endswith(('.png', '.jpg'))]
    for img_filename in imgs_filenames:
        img = imageio.imread(join(img_base_path, img_filename))
        img = transforms.ToTensor()(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
        img = img.numpy().reshape((800, 800, 3)) * 255

        img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.uint8)
        imageio.imsave(join(saving_path, img_filename), img)
