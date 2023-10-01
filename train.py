import ast
from PIL import Image
from torch.autograd import Variable
from torch import __version__
import argparse
import torch
import numpy as np
import os
from torch import nn,optim
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from torchvision import models
from torchvision import datasets, transforms

# 
def get_input_args():
    """
     
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data', type = str, default = 'flowers', 
                    help = 'image training data')
    
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                    help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'Choose architecture from vgg, alexnet and resnet')
    parser.add_argument('--learning_rate', type = str, default = '0.001', help = 'Set hyperparameter: Learning rate')
    parser.add_argument('--hidden_units', type = str, default = '512', help = 'Set hyperparameter: Hidden units')
    parser.add_argument('--epochs', type = str, default = '5', help = 'Set hyperparameter: Epcochs')
    parser.add_argument('-gpu','--gpu', action='store_true' , help='Use GPU for training')
    
    # you created with this function 
    return parser.parse_args()

in_arg = get_input_args()

# resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = { 'alexnet': [alexnet,9216], 'vgg': [vgg16,25088]}
# models = {'vgg16': vgg16}


data_dir = in_arg.data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test' #dataloader
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),      
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

val_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
valloader = torch.utils.data.DataLoader(train_data, batch_size=64)


model = models[in_arg.arch][0]
model.class_to_idx = train_data.class_to_idx
# Use GPU if it's available
if (in_arg.gpu & torch.cuda.is_available()):
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
print(f"You are going to use {device} for training!")
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(models[in_arg.arch][1],int(in_arg.hidden_units)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
#                                  nn.Linear(4096, 4096),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.5),
                                 nn.Linear(int(in_arg.hidden_units),102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=float(in_arg.learning_rate))

model.to(device);
epochs = int(in_arg.epochs)
steps = 0
running_loss = 0
print_every = 32
print("Training process is started...")
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. | "
                  f"Train loss: {running_loss/print_every:.3f} | "
                  f"Test loss: {test_loss/len(valloader):.3f} | "
                  f"Test accuracy: {accuracy/len(valloader):.3f}")
            running_loss = 0
            model.train()
print('Training process is success! ')
model.to("cpu")
#creating dictionary 
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'mapping':    model.class_to_idx
             } 

torch.save(checkpoint, in_arg.save_dir)


