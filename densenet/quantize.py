'''
loads a trained model and carries out INQ
'''

import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim

from densenet import DenseNet
from data import trainloader, testloader, validation_loader
from inq.quantization_scheduler import INQScheduler
from inq.sgd import SGD
from common import reset_lr_scheduler

from types import SimpleNamespace
import datetime
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = DenseNet(bn_size=6, drop_rate=0.1, block_config=(36,36,36), growth_rate=16)
net.to(device)

checkpoint = torch.load('saved/a624e5f1-cc56-477d-ba87-837e6072d309.tar')
net.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss().to(device)
optimizer = SGD(net.parameters(), lr=0.1, weight_decay=1e-4, nesterov=True, momentum=0.9)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

quantization_settings_dict = {
    'iterative_steps': [0.25, 0.5, 0.75, 0.875, 1],
    'epochs': 8 # retrain epochs
}

def validate(model, loader, criterion):
    start_t = datetime.datetime.now()
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        validation_loss += loss.item()

    validation_accuracy = correct / total
    print("time %s, validation loss: %.3f, validation accuracy %.4f %%" %
          (str(datetime.datetime.now() - start_t), validation_loss, validation_accuracy * 100))

    return validation_loss, validation_accuracy

def quantize(settings_dict):
    settings = SimpleNamespace(**settings_dict)
    # validation error before quantization
    # validate(net, validation_loader, criterion)
    # start quantization
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2])
    quantization_scheduler = INQScheduler(optimizer, settings.iterative_steps, weight_bits=5)
    for campaign in range(len(settings.iterative_steps) - 1):
        print("campaign:", campaign)
        # reset_lr_scheduler(scheduler)
        quantization_scheduler.step()
        for epoch in range(settings.epochs):
            print("epoch: ", epoch)
            # train
            t1 = datetime.datetime.now()
            net.train()
            train_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print("train loss: %.3f, took %s" % (train_loss, str(datetime.datetime.now() - t1)))
            # scheduler.step()
        print("test loss: ")
        validate(net, testloader, criterion)
    # quantize all remaining weights
    quantization_scheduler.step()
    # test loss
    print("test loss:")
    validate(net, testloader, criterion)
    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, 'saved/' + "quantized2.tar")
    

def main():
    quantize(quantization_settings_dict)
    

if __name__ == '__main__':
    main()
