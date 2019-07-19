TRANSFORM_NORMALIZATION = ((0.53129727, 0.5259391, 0.52069134), (0.28938246, 0.28505746, 0.27971658))
BATCH_SIZE = 48

# global variables
VALIDATION_SPLIT = 0.05
EPOCHS = 300

import torch
import torchvision
import torchvision.transforms as transforms

# load data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*TRANSFORM_NORMALIZATION)])

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'), # add whitespace
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*TRANSFORM_NORMALIZATION)])

trainset = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=train_transform)

# split validation
train_length = len(trainset)
validation_length = int(VALIDATION_SPLIT * train_length)
train_length -= validation_length
[trainset, validation_set] = torch.utils.data.random_split(
    dataset=trainset, lengths=[train_length, validation_length])
print("trainset length: %d, validation set length: %d" %
      (len(trainset), len(validation_set)))

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=2*BATCH_SIZE,
    shuffle=True,
    num_workers=2)

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=test_transform)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2)
