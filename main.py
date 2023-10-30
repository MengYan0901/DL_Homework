import argparse
import torch
from models.model import Unet
from models.gaussian_diffusion2 import GaussianDiffusion
from dataset import get_data
from models.gaussian_diffusion2 import get_loss
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

    train_data, val_data = get_data(args, return_dataset=False)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )

    from torch.optim import Adam

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100  # Try more!
    T = 300

    # # Use wandb
    # key = '1bed216d1f9c32afa692155d2e0911cd750f41dd'
    # wandb.login(key=key)
    #
    # # start a new experiment
    # config = dict(
    #     dataset=args.dataset, batch_size=args.batch_size, architecture='Unet'
    # )
    #
    # wandb.init(project="diffusion-model", name=args.name_exp, config=config,
    #            entity='deep-learning-home-work')

    for epoch in range(epochs):
        avg_loss = 0
        for step, batch in enumerate(train_data):
            optimizer.zero_grad()

            if batch[0].shape[0] == args.size:  # last batch will cause problem because of the batch_size

                t = torch.randint(0, T, (args.size,), device=device).long()
                loss, denoised_image = get_loss(model, batch[0], t, device)
                loss.backward()
                optimizer.step()

                avg_loss += loss
                # if epoch % 1 == 0 and step == 0:
                #     print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

                if epoch % 1 == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

                    clean_image_cpu = batch[0].cpu().detach().numpy()
                    denoised_image_cpu = denoised_image.cpu().detach().numpy()

                    # print(clean_image_cpu.shape, denoised_image_cpu.shape)
                    # Calculate PSNR
                    psnr = peak_signal_noise_ratio(clean_image_cpu, denoised_image_cpu)

                    # # Calculate SSIM
                    # ssim = structural_similarity(batch[0].cpu().detach().numpy(),
                    #                              denoised_image.cpu().detach().numpy(), multichannel=True)

                    # Print the results
                    print(f"PSNR: {psnr:.2f} dB")
                    # print(f"SSIM: {ssim:.4f}")

                    # image_array = sample_plot_image(args, model)
                    #
                    # images = wandb.Image(
                    #     image_array,
                    #     caption=f"Sample image at {epoch}"
                    # )
                    #
                    # wandb.log({"Sample images": images})
                    #
                    # wandb.log({'loss': loss, 'epoch': epoch})


if __name__ == '__main__':
    main()
