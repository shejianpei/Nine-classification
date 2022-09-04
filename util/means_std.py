import numpy as np
import cv2
import random
import os
from PIL import Image

# calculate means and std  注意换行\n符号**
# train.txt中每一行是图像的位置信息**
path = 'E:/佘哥的青春/大二下/舌象分析/nine_classification/util/image_list.txt'
means = [0, 0, 0]
stdevs = [0, 0, 0]

index = 1
num_imgs = 0
with open(path, 'r') as f:
    lines = f.readlines()
    # random.shuffle(lines)

    for line in lines:
        print(line)
        print('{}/{}'.format(index, len(lines)))
        index += 1
        a = os.path.join(line)

        # print(a[:-1])
        num_imgs += 1
        img = Image.open(a[:-1])
        print(img)
        img = np.asarray(img)

        img = img.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
print(num_imgs)
means.reverse()
stdevs.reverse()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

# # 求数据集的均值和标准差
# data_transform = transforms.transforms.Compose([
#     transforms.Resize((32, 32)),    # 缩放到32 * 32
#     transforms.RandomCrop(32, padding=4),    # 裁剪
#     transforms.ToTensor(),    # 转化为张量
# ])
#
# train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform)
# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
# mean = torch.zeros(3)
# std = torch.zeros(3)
# print('==> Computing mean and std..')
# for inputs, targets in dataloader:
#     for i in range(3):
#         # print(inputs)
#         mean[i] += inputs[:, i, :, :].mean()
#         std[i] += inputs[:, i, :, :].std()
# mean.div_(len(train_dataset))
# std.div_(len(train_dataset))
# print("mean:{} and std: {}".format(mean,std))

# 后续调参时可直接使用求出的标准差和方差