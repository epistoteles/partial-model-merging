from src.utils import *
from src.evaluate import *
model_a, model_b = load_models_ab('CIFAR10-VGG11-1x')
train_aug_loader, train_noaug_loader, test_loader = get_loaders('CIFAR10')
