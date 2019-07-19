import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from naive_net import NaiveNet
from mobilenetv2 import MobileNetV2
from densenet import DenseNet

from data import *

import uuid
import datetime
model_name = str(uuid.uuid4())

print("model name is ", model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

print(device)

# train
# net = NaiveNet()
# net = MobileNetV2(n_classes=100)
net = DenseNet(bn_size=6, drop_rate=0.1, block_config=(36,36,36), growth_rate=16)
net.to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4, nesterov=True, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*EPOCHS), int(0.75*EPOCHS)], gamma=0.1)

for epoch in range(EPOCHS):

    running_loss = 0.0
    net.train(True)
    for i, data in enumerate(trainloader, 0):  # 40000/32
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    # validation
    validation_loss = 0.0
    correct = 0
    total = 0
    net.train(False)
    for i, data in enumerate(validation_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        validation_loss += loss.item()

    print("time %s, epoch %d, loss: %.3f, validation loss: %.3f, validation accuracy %.4f %%" %
          (str(datetime.datetime.now()) ,epoch+1, running_loss, validation_loss, correct / total * 100))

    validation_accuracy = correct / total
    scheduler.step()

print("Finished training")

# test accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # top 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Top1 accuracy of the network on the 10000 test images: %.4f %%' %
      (100 * correct / total))

# save model

torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss
}, 'saved/' + model_name + '.tar')

print("Model saved ", model_name)
