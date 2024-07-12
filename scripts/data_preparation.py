import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import yaml

def load_config(config_path):
    """Load configuration from a yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config):
    """Prepare the dataset based on the configuration."""
    data_config = config['data']
    transform = transforms.Compose([
        transforms.RandomResizedCrop(data_config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if data_config['dataset'].lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/raw', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {data_config['dataset']} is not supported")

    os.makedirs('./data/processed', exist_ok=True)
    torch.save(train_dataset, './data/processed/train_dataset.pt')
    torch.save(test_dataset, './data/processed/test_dataset.pt')

def main():
    config = load_config('config.yaml')
    prepare_data(config)
    print("Data preparation completed successfully.")

if __name__ == "__main__":
    main()
