import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(image_dir: str):
    
    """Get mean and std."""
    dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=image_dir, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])), batch_size=4
    )
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    count = 0
    for data, _ in tqdm(dataloader):

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
        count +=1
        if count%100==0:
            mean = channels_sum / num_batches
            std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
            print(f'count->{count}, mean->{mean}, std->{std}')
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    args = parser.parse_args()
    main(
        image_dir=args.image_dir,
    )
