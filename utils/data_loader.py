import yaml
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_name = config['data']['dataset']
image_size = config['data']['image_size']

def get_datasets():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='../data/raw', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='../data/raw', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    return train_dataset, test_dataset
