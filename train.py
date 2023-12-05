#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
from PIL import ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from smdebug import modes
from smdebug.pytorch import get_hook

import argparse
import os


def test(model, testloader, cost_function, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    hook = get_hook(create_if_not_exists=True)
    hook.set_mode(modes.EVAL)
    
    with torch.no_grad():
        for data, label in testloader:
            data = data.to(device)
            label = label.to(device)
            prediction = model(data)
            running_loss += cost_function(prediction, label).item() * data.size(0)
            categories = torch.argmax(prediction, axis=1)
            running_corrects += torch.sum(categories==label.data)
        
    print(f"Test loss: {running_loss/len(testloader.dataset):.4f}, test accuracy: {running_corrects*1.0/len(testloader.dataset)*100:.4f}%")    

def train(model, dataloaders, criterion, optimizer, device, epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = get_hook(create_if_not_exists=True)
    hook.register_loss(criterion)
    
    for e in range(epochs):
        print(f'----------Epoch {e}----------')
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            
            if phase == 'train':
                model.train()
                torch.set_grad_enabled(True)
                hook.set_mode(modes.TRAIN)
            else:
                model.eval()
                torch.set_grad_enabled(False)
                hook.set_mode(modes.EVAL)
                
            for data, label in dataloaders[phase]:
                data = data.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                
                preds = model(data)
                categories = torch.argmax(preds, axis=1)
                loss = criterion(preds, label)
                if phase == 'train':
                    # update weights
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(categories == label.data)
                
            print(f"{phase} loss: {running_loss/len(dataloaders[phase].dataset):.4f}, {phase} accuracy: {running_corrects*1.0/len(dataloaders[phase].dataset)*100:.4f}%") 

    return model

            
def net(model_type):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    if model_type == 'ResNet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)
    elif model_type == 'ResNet34':
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)
    else:
        print('Not a valid model type!')
    return model

def create_data_loaders(data_path, batch_size, transform):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model=net(args.model_type)
    model = model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # create data loaders
    data_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                         transforms.ToTensor(), 
                                         transforms.Normalize([0.52, 0.44, 0.35], [0.15, 0.14, 0.105])])
    train_loader = create_data_loaders(args.train, args.batch_size, data_transform)
    val_loader = create_data_loaders(args.val, 1000, data_transform)
    test_loader = create_data_loaders(args.test, 1000, data_transform)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, dataloaders, loss_criterion, optimizer, device, args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_type", type=str, default='ResNet18')
    
    parser.add_argument("--train", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--val", type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument("--test", type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--model_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    main(args)
