import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

loss_func = nn.CrossEntropyLoss()

def calculate_accuracy(fx,y):
    preds = fx.max(1,keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def train(model,device,iterator):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for (x,y) in iterator:
        x = x.to(device)
        y = y.to(device)

        fx = model(x)

        optimizer.zero_grad()
        loss = loss_func(fx,y)
        loss.backward()
        scheduler.step(1)
        optimizer.step()

        acc = calculate_accuracy(fx,y)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator),epoch_acc/(len(iterator))

def evaluate(model,device,iterator):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x,y) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            _, predicted = torch.max(fx.data, 1)

            loss = loss_func(fx, y)
            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / (len(iterator))

