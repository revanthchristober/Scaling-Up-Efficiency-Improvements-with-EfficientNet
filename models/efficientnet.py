# models/efficientnet.py
from efficientnet_pytorch import EfficientNet

def get_efficientnet_model(version='efficientnet-b0'):
    model = EfficientNet.from_pretrained(version)
    return model
