from torchvision.transforms import v2
import numpy as np
import torch


def show_tensor_image(image):
    reverse_transforms = v2.Compose([
        v2.Lambda(lambda t: (t + 1) / 2),
        v2.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        v2.Lambda(lambda t: t * 255.),
        v2.Lambda(lambda t: t.to(torch.float32)),
        # v2.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    return reverse_transforms(image)


@torch.no_grad()
def sample_plot_image(args, model):
    from models.gaussian_diffusion2 import T, sample_timestep

    device = 'cuda'
    # Sample noise
    img_size = args.size
    img = torch.randn((1, 3, img_size, img_size), device=device)

    num_images = 10
    stepsize = int(T/num_images)

    image_array = []
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

        reversed_image = show_tensor_image(img)
        image_array.append(reversed_image)

    return image_array

