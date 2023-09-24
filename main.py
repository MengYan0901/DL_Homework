import argparse
import torch
from models.model import Unet
from models.gaussian_diffusion2 import GaussianDiffusion
from dataset import get_data


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--size', type=int, default=64, help='Image size')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data = get_data(args, return_dataset=False)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=1000  # number of steps
    )

    for i, data in enumerate(train_data):
        img, lable = data

        loss = diffusion(img)

        print(loss)

        if i==0:
            break


if __name__ == '__main__':
    main()