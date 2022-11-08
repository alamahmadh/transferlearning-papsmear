import os
import copy
import time
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models, transforms 

from config import Config
from data import PapSmearDataset
from models import *


def params_to_update(model_ft, feature_extract):
    '''
    Collect parameters to be updated (unfrozen)
    
    Feature extracting is to create an optimizer that only updates the desired parameters.
    This only works if the layer(s) is set to have requires_grad=True

    '''
    params_to_update = model_ft.parameters()
    
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    
    return params_to_update

def train(model, dataloaders, criterion, optimizer, num_epochs):
    start = time.time()
    val_acc_history = []
    
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 20)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() #set model to training mode
            else:
                model.eval() #set model to eval mode
            
            running_loss = 0.0
            running_corrects = 0
            
            for data in dataloaders[phase]:
                X, y = data.values()
                X = X.to(device)
                y = y.to(device)
                
                #zero the parameter gradients
                optimizer.zero_grad()
                
                #track history if "train"
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    #backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                #statistics
                running_loss += loss.item() * X.size(0) #the averaged accumulative loss in a batch
                running_corrects += torch.sum(preds == y.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f"{phase} loss: {epoch_loss:.4f} & acc: {epoch_acc:.4f}")
            
            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #the new weights are saved once the accuracy improves in an epoch
                #otherwise, the old weights will be used
                best_weights = copy.deepcopy(model.state_dict()) 
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
    time_elapsed = time.time() - start
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")
        
    #load best model weights
    model.load_state_dict(best_weights)
        
    return model, val_acc_history

def save_model(model_ft, model_path, model_name, epochs):
    model_file = os.path.join(model_path, model_name + '_' + str(epochs) + '.pth')
    torch.save(model_ft.state_dict(), model_file)
    return model_file

if __name__ == "__main__":
    
    pars = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Data transform
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(pars.mean, pars.std)
        ]),
        'val': transforms.Compose([
            #transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(pars.std, pars.std)
        ]),
    }
    
    #Define dataset and dataloaders
    image_datasets = {x: PapSmearDataset(
        os.path.join(pars.data_root, 'csv_'+ x +'.csv'), 
        data_transforms[x]
    ) for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(
        image_datasets[x], 
        batch_size=pars.batch_size, 
        shuffle=True, 
        num_workers=pars.num_workers
    ) for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].find_labels()
    num_classes = len(class_names)
    
    #load the pretrained model
    model_ft = None
    input_size = 0
    
    if pars.model == "resnet18":
        model_ft, input_size = resnet18(num_classes, use_pretrained=True)
    elif pars.model == "squeezenet":
        model_ft, input_size = squeezenet(num_classes, use_pretrained=True)
    else:
        print("Invalid model, please define a new model in models.py")
        
    model_ft = model_ft.to(device)
    if pars.optimizer == 'sgd':
        optimizer_ft = optim.SGD(params_to_update(model_ft, pars.feature_extract), lr=pars.lr)
    elif pars.optimizer =="adam":
        optimizer_ft = optim.Adam(params_to_update(model_ft, pars.feature_extract), lr=pars.lr)
    
    criterion = nn.CrossEntropyLoss()
    
    #Train model
    model_ft, hist = train(
        model_ft, 
        dataloaders, 
        criterion, 
        optimizer_ft, 
        num_epochs=pars.num_epochs
    )
    
    save_model(model_ft, pars.model_path, pars.model, pars.num_epochs)
