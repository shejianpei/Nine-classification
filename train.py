import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from resnet.net import net
import torchvision.models as models

from imgPretreatment.imgPre import imgpre
from util.funcs import *

import os
import random
import numpy as np


device = torch.device('cuda')
train_iterator,valid_iterator = imgpre()
model = net()


EPOCHS = 10
SAVE_DIR='models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR,'resnet18-tongue.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(EPOCHS):
    train_loss,train_acc = train(model,device,train_iterator)
    valid_loss,valid_acc = evaluate(model,device,valid_iterator)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),MODEL_SAVE_PATH)

    print(F'|Epoch:{epoch+1:02} | Train_loss:{train_loss:.3f} | Train Acc:{train_acc*100:05.2f}% | Val.Loss:{valid_loss:.3f} | Val.Acc:{valid_acc*100:05.2f}%')



