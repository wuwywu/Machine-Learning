# 自定义方法处理数据集
# wuyong@ccnu
# 手写数字数据集下载网址MNIST：http://yann.lecun.com/exdb/mnist/index.html
# train-images-idx3-ubyte.gz: 60000个训练图片
# train-labels-idx1-ubyte.gz: 60000个训练标签
# t10k-images-idx3-ubyte.gz: 10000个测试图片
# t10k-labels-idx1-ubyte.gz: 10000个测试标签

import urllib.request
import gzip
import os,time
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch

class MnistDataset:
    def __init__(self, trian =True, root='./datas', normalize=True, flatten=False):
        # 数据集地址
        self.url_base = r'http://yann.lecun.com/exdb/mnist/'

        # 创建文件夹
        if not os.path.exists(root):    os.makedirs(root)

        # 归一化判断条件
        self.normalize = normalize

        # 扁平化条件
        self.flatten = flatten

        # 文件名数组
        key_file = {
            'train_img':'train-images-idx3-ubyte.gz',
            'train_label':'train-labels-idx1-ubyte.gz',
            'test_img':'t10k-images-idx3-ubyte.gz',
            'test_label':'t10k-labels-idx1-ubyte.gz'
        }

         # 载入和下载数据集
        if trian:  
            # 数据地址
            filename1 = root+"/"+key_file['train_img'][:-3]
            filename2 = root+"/"+key_file['train_label'][:-3]
            # 下载路径
            file_download = root+"/"+"download"+"/"       
            if not os.path.exists(filename1):
                if not os.path.exists(file_download):    os.makedirs(file_download)
                # 数据下载地址
                filename1_download = file_download+key_file['train_img']
                # 下载
                self._download(key_file['train_img'], filename1_download)
                # 解包
                self._gzip(filename1, filename1_download)
            if not os.path.exists(filename2):
                if not os.path.exists(file_download):    os.makedirs(file_download)
                # 数据下载地址
                filename2_download = file_download+key_file['train_label']
                # 下载
                self._download(key_file['train_label'], filename2_download)
                # 解包
                self._gzip(filename2, filename2_download)
        elif not trian:
            # 数据地址
            filename1 = root+"/"+key_file['test_img'][:-3]
            filename2 = root+"/"+key_file['test_label'][:-3]
            # 下载路径
            file_download = root+"/"+"download"+"/" 
            if not os.path.exists(filename1):
                if not os.path.exists(file_download):    os.makedirs(file_download)
                # 数据下载地址
                filename1_download = file_download+key_file['test_img']
                # 下载
                self._download(key_file['test_img'], filename1_download)
                # 解包
                self._gzip(filename1, filename1_download)
            if not os.path.exists(filename2):
                if not os.path.exists(file_download):    os.makedirs(file_download)
                # 数据下载地址
                filename2_download = file_download+key_file['test_label']
                # 下载
                self._download(key_file['test_label'], filename2_download)
                # 解包
                self._gzip(filename2, filename2_download)

        else: print("train应该是布尔是数据类型")

        # 读取数据集
        self._read_dataset_fromPath(filename1, filename2)

    def _read_dataset_fromPath(self, filename1, filename2):
        if os.path.exists(filename1):
            with open(filename1, 'rb') as imgpath:
                print("dataset_images is existing")
                self.images_magic, self.images_num, self.rows, self.cols = struct.unpack('>IIII', imgpath.read(16))
                self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(self.images_num, self.rows * self.cols)
        else:   print("dataset_images is inexisting")

        if os.path.exists(filename2):
            print("dataset_labels is existing")
            with open(filename2, 'rb') as lbpath:
                self.labels_magic, self.labels_num = struct.unpack('>II', lbpath.read(8))
                self.labels = np.fromfile(lbpath, dtype=np.uint8)
        else:   print("dataset_labels is inexisting")

    def _download(self, file_name, root):
        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(self.url_base + file_name, root) # 下载
        print("Done")

    def _gzip(slef, filename, filename_download):
        with gzip.open(filename_download, 'rb') as f1:
            with open(filename, 'wb') as f2:
                f2.write(f1.read())

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # 目标图像 (标签)
        label = self.labels[index]
        size = label.size   # 批次大小   
        if size==1:           
            target = torch.zeros((10))
            target[label] = 1.0
        else:
            target = torch.zeros((label.size, 10))
            for idx, row in enumerate(label):
                target[idx, row] = 1.0

        # 图像数据, 取值范围是0~255，标准化为0.01~1
        if self.normalize:  image_values = (torch.FloatTensor(self.images[index]))/255*0.99+0.01

        if not self.flatten:     image_values = (torch.FloatTensor(self.images[index])).reshape(-1, 1, 28, 28)

        return label, image_values, target
    
    def plot_image(self, choose_num):
        label = self.labels[choose_num]
        image = self.images[choose_num].reshape(28,28)
        # 交互（interactive）模式。即使在脚本中遇到plt.show()，代码还是会继续执行
        # plt.ion()
        figure, ax = plt.subplots()
        im = ax.imshow(image)
        plt.colorbar(im)
        plt.title('the label is : {}'.format(label))
        plt.show()
        # time.sleep(100)

    def run_images(self):

        # 交互（interactive）模式。即使在脚本中遇到plt.show()，代码还是会继续执行
        plt.ion()
        label = ""
        image = np.random.randint(0, 255, size=(self.rows, self.cols))
        figure, ax = plt.subplots()
        im = ax.imshow(image)
        # plt.colorbar(im)
        plt.title('the label is : {}'.format(label))
        # plt.show()

        for choose_num in range(0, self.images_num):
            # print(choose_num)
            label = self.labels[choose_num]
            image = self.images[choose_num].reshape(28,28)
            # 画图更新
            im.set_data(image)  # update image data
            # draw and flush the figure
            plt.title('the label is : {}'.format(label)) # 图像题目
            figure.canvas.draw()
            figure.canvas.flush_events()
            # 等待
            time.sleep(1)
        plt.show()

    def debug(self):
        print('labels_magic is {} \n'.format(self.labels_magic),
        'labels_num is {} \n'.format(self.labels_num),
        'labels is {} \n'.format(self.labels))

        print('images_magic is {} \n'.format(self.images_magic),
        'images_num is {} \n'.format(self.images_num),
        'rows is {} \n'.format(self.rows),
        'cols is {} \n'.format(self.cols),
        'images is {} \n'.format(self.images))


if __name__=="__main__":
    Mnist = MnistDataset(trian=True)        # flatten=False

    # Mnist.plot_image(0)
    # Mnist.run_images()
    # print(len(Mnist))
    # index = 3
    # for label, image_values, target in Mnist:
    #     print(label)
    label, image_values, target = Mnist[4:6]
    # target[0:3] = 1
    print(label)

    print(target)

    print(image_values.shape)

    # os.makedirs(r"./MnistDataset")
    # print(os.path.exists(r"./MnistDataset"))
