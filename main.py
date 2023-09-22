import argparse
import torch

from dataset import get_data


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data = get_data(args, return_dataset=True)


if __name__ == '__main__':
    main()