import os


def generate(dir, label):
    listText = open('list.txt', 'a')
    for file in dir:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()


outer_path = 'E:/佘哥的青春/大二下/舌象分析/data/'  # 这里是你的图片的目录

if __name__ == '__main__':
    i = 1
    num = 0
    datalist = os.listdir(outer_path)  # 列举文件夹  train
    datalist.sort()
    for train_test_vail in datalist:
        animal = outer_path + train_test_vail + "/"

        animallist = os.listdir(animal)   #cat
        animallist.sort()

        for cat_dog in animallist:
            catPath = outer_path + train_test_vail + "/" + cat_dog+"/"
            catlist = os.listdir(catPath)
            catlist.sort()

            for final in catlist:

                finallPATH = os.path.join(outer_path, train_test_vail, cat_dog,final)

                finallPATH = finallPATH.replace('\\', '/')

                listText = open('image_list.txt', 'a')
                fileType = os.path.split(finallPATH)

                name = finallPATH + '\n'

                listText.write(name)

            listText.close()
