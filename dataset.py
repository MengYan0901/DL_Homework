# Import dataset
from torchvision.datasets import CelebA, Flowers102
from torch.utils.data import DataLoader

# Transform
import torchvision.transforms as transforms


def celebA_transform():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a common size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1]
    ])
    target_transform = None
    return transform, target_transform


def flowers102_transform(args):
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),  # Resize images to a common size (adjust as needed)
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally for data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet mean and standard deviation
    ])
    target_transform = None  # You can define a target transform if needed
    return transform, target_transform


def get_data(args, return_dataset=False):
    root_data_path = '/home/luu/DeepLearning_HW/datasets'
    if args.dataset == 'celebA':
        transform, target_transform = celebA_transform()
        train_dataset = CelebA(root=root_data_path, split='train', target_type='attr',
                               transform=transform, target_transform=target_transform, download=True)

        val_dataset = CelebA(root=root_data_path, split='val', target_type='attr',
                             transform=transform, download=False)

    elif args.dataset == 'flowers102':
        transform, target_transform = flowers102_transform(args)
        train_dataset = Flowers102(root=root_data_path, split='train',
                                   transform=transform, target_transform=target_transform, download=True)

        val_dataset = Flowers102(root=root_data_path, split='val',
                                 transform=transform, download=False)
    else:
        train_dataset = None
        val_dataset = None
        raise Exception("The dataset is not defined")

    if return_dataset:
        return train_dataset, val_dataset
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        return train_dataloader, val_dataloader
