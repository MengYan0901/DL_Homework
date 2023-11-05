import argparse
import torch
from models.model import Unet
from dataset import get_data
from models.gaussian_diffusion2 import GaussianDiffusion, Trainer
import wandb
from utils import sample_plot_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--size', type=int, default=32, help='Image size')
    parser.add_argument('--name_exp', type=str, default='experiment_diffusion_model', help='Describe the experiment')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data = get_data(args, return_dataset=True)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=1000  # number of steps
    )

    trainer = Trainer(
        diffusion,
        train_data,
        train_batch_size=args.batch_size,
        train_lr=8e-5,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True  # whether to calculate fid during training
    )

    trainer.train()


if __name__ == '__main__':
    main()
