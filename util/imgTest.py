import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from resnet.net import net
from util.funcs import evaluate
from imgPretreatment.imgPre import *
from util.funcs import *
import cv2

MODEL_PATH = './models/resnet18-tongue.pt'

test_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.21740138,0.21138199 ,0.29182663), (0.26943176 ,0.26165729 ,0.35083898))

    ])

def imgtest(imgpath):

    device = torch.device('cuda')
    model = net()
    model.load_state_dict(torch.load(MODEL_PATH))



    BATCH_SIZE = 1
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        model = model.to(device)
        test_data = Portrait_dataset(imgpath, test_transforms)
        # print(f'Number of testing examples: {len(test_data)}')
        test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

        loss_func = nn.CrossEntropyLoss()



        with torch.no_grad():
            for (x, y) in test_iterator:
                x = x.to(device)
                y = y.to(device)

                fx = model(x)

                _, predicted = torch.max(fx.data, 1)

                loss = loss_func(fx, y)
                acc = calculate_accuracy(fx, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if predicted[0] == 0:
                    rs = 'pinghe'
                elif predicted[0] == 1:
                    rs = 'qixu'
                elif predicted[0] == 2:
                    rs = 'qiyu'
                elif predicted[0] == 3:
                    rs = 'shire'
                elif predicted[0] == 4:
                    rs = 'tanshi'
                elif predicted[0] == 5:
                    rs = 'tebing'
                elif predicted[0] == 6:
                    rs = 'xueyu'
                elif predicted[0] == 7:
                    rs = 'yangxu'
                else:
                    rs = 'yinxu'

                print("模型分类结果{} 实际结果{}".format(rs, y))

        test_loss = epoch_loss / len(test_iterator)
        test_acc = epoch_acc / (len(test_iterator))

        print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:05.2f}% |')

def imgtest_1(imgpath):

    device = torch.device('cuda')
    model = net()
    model.load_state_dict(torch.load(MODEL_PATH))

    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        model = model.to(device)
        img_r = cv2.imread(imgpath)  # 读入图片
        img = Image.open(imgpath)
        img_tensor = test_transforms(img)
        img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to("cuda")

        fx = model(img_tensor)
        _, predicted = torch.max(fx.data, 1)

        if predicted[0] == 0:
            rs = 'pinghe'
        elif predicted[0] == 1:
            rs = 'qixu'
        elif predicted[0] == 2:
            rs = 'qiyu'
        elif predicted[0] == 3:
            rs = 'shire'
        elif predicted[0] == 4:
            rs = 'tanshi'
        elif predicted[0] == 5:
            rs = 'tebing'
        elif predicted[0] == 6:
            rs = 'xueyu'
        elif predicted[0] == 7:
            rs = 'yangxu'
        else:
            rs = 'yinxu'

        print("模型分类结果{} ".format(rs))



