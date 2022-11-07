import torch
import torch.nn as nn
from torchvision import models

'''
if we are feature extracting and only want to compute gradients for the newly initialized layer 
then we want all of the other parameters to not require gradients. 

'''

def set_parameters_requires_grad(model, feature_extract=True):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
'''
The final layer of a CNN model is often an FC layer, 
has the same number of nodes as the number of output classes in the dataset.

The goal here is to reshape the last layer to have the same number of inputs as before, 
AND to have the same number of outputs as the number of classes in the dataset. 

'''

def resnet18(num_classes, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameters_requires_grad(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    
    return model_ft, input_size

def squeezenet(num_classes, use_pretrained=True):
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameters_requires_grad(model_ft)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes
    input_size = 224
    
    return model_ft, input_size