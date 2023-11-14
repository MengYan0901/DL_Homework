import argparse
from models.gaussian_diffusion2 import GaussianDiffusion
import torch
from models.model import Unet
import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--size', type=int, default=32, help='Image size')
    parser.add_argument('--model_weights_path', type=str, default='/data2/Users/mengke/unet_model_state.pth',
                        help='model weights path')
    parser.add_argument('--output_save_path', type=str, default='/home/mengke/DL_Homework/output_image.png',
                        help='The path of saving output image')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.size,
        timesteps=10000)  # number of steps

    model.load_state_dict(torch.load(args.model_weights_path))

    sampled_images = diffusion.sample(batch_size=args.batch_size)
    sampled_images = sampled_images.cpu().detach().numpy()
    print(sampled_images)

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))

    for i, img in enumerate(sampled_images):
        ax = axes[i]
        ax.imshow(img.transpose(1, 2, 0))
        ax.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(args.output_save_path)
    plt.show()
    print('ok')


if __name__ == '__main__':
    main()