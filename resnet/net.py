import torchvision.models as models
import torch.nn as nn
import torch

device = torch.device('cuda')
def net():
    model = models.resnet18(pretrained=True).to(device)
    #print(model.fc)
    #Linear(in_features=512, out_features=1000, bias=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features=512, out_features=9).to(device)
    return model

net()