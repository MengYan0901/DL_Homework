import argparse
import torch
from dataset import get_data
import json
import matplotlib.pyplot as plt


def show_random_original_images(args, dataset, num_images, cols):

    if args.dataset == 'flowers102':
        # Specify the path to your JSON file
        json_file_path = "/home/luu/DeepLearning_HW/flower_to_name.json"

        # Open the JSON file for reading
        with open(json_file_path, "r") as json_file:
            # Load the JSON data from the file
            label_names = json.load(json_file)

    cols = cols
    rows = (num_images + cols - 1) // cols

    # Create a figure to display images
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, num_images + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]

        if str(label) in label_names:
            figure.add_subplot(rows, cols, i)
            plt.title(label_names[str(label)])
            plt.axis("off")
            plt.imshow(img.squeeze().permute(1, 2, 0))

    print("saving original images")
    visualize_path = '/home/luu/DeepLearning_HW/results/visualize/'
    plt.savefig(visualize_path + f"{num_images}_sample_images.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--size', type=int, default=64, help='Image size')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, val_data = get_data(args, return_dataset=True)

    # Visualize original data
    show_random_original_images(args, train_data, 10, 5)
    # Remember to check the transforms in the function get_data to get the original images or not


if __name__ == '__main__':
    main()
