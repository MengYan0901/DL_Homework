import argparse
import torch
from models.model import Unet
from models.gaussian_diffusion2 import GaussianDiffusion
from dataset import get_data
from models.gaussian_diffusion2 import get_loss
import wandb


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--size', type=int, default=64, help='Image size')
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

    # Use wandb
    key = '1bed216d1f9c32afa692155d2e0911cd750f41dd'
    wandb.login(key=key)

    # start a new experiment
    config = dict(
        dataset=args.dataset, batch_size=args.batch_size, architecture='Unet'
    )

    wandb.init(project="diffusion-model", name=args.name_exp, config=config,
               entity='deep-learning-home-work')

    for epoch in range(epochs):
        for step, batch in enumerate(train_data):
            optimizer.zero_grad()

            if batch[0].shape[0] == args.size:  # last batch will cause problem because of the batch_size
                t = torch.randint(0, T, (args.size,), device=device).long()
                loss = get_loss(model, batch[0], t, device)
                loss.backward()
                optimizer.step()

                if epoch % 1 == 0 and step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")


if __name__ == '__main__':
    main()
