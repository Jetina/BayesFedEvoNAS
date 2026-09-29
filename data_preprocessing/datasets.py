import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets import MNIST, CIFAR10
from torchvision.datasets import FashionMNIST

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        # 加载完整的数据集
        self.fashion_mnist_dataobj = FashionMNIST(self.root, self.train, transform=None, target_transform=None, download=self.download)

        # 根据索引切分数据集
        if self.train:
            self.data = self.fashion_mnist_dataobj.data
            self.targets = self.fashion_mnist_dataobj.targets
        else:
            self.data = self.fashion_mnist_dataobj.data
            self.targets = self.fashion_mnist_dataobj.targets

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        #用来加载和预处理MNIST数据集的数据
        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        '''
        使用当前实例的train属性来决定是否从MNIST对象中加载训练集或测试集。
        由此产生的数据和目标变量被分配给MNIST对象的相应属性。
        '''
        if self.train:
            data = mnist_dataobj.train_data
            target = mnist_dataobj.train_labels
        else:
            data = mnist_dataobj.test_data
            target = mnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')#图像的numpy数组转换为PIL图像对象,.numpy转成numpy类型，L表示他是个灰度图像
        #转换成PIL有助于进行转换，比如裁剪等

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self. data)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=True):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    '''
    有选择地将对象的数据属性的某些索引的第二和第三通道设置为0，从而 "截断 "这些通道。
    '''
    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR10ColorGrayScale(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self.download = download
        self._gray_scale_indices = []

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        if self.train:
            #print("train member of the class: {}".format(self.train))
            #data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        self._gray_scale_indices = index#属性名称前的下划线表示它是一个私有属性，不应该被对象之外的代码直接访问或修改。
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = self.data[gs_index, :, :, 0]#第一个通道的值被分配给第二个和第三个通道，有效地使它们成为灰度
            self.data[gs_index, :, :, 2] = self.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)



class CIFAR10ColorGrayScaleTruncated(data.Dataset):
    def __init__(self, root, dataidxs=None, gray_scale_indices=None,
                    train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self._gray_scale_indices = gray_scale_indices
        self.download = download

        self.cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        # we need to trunc the channle first
        self.__truncate_channel__(index=gray_scale_indices)
        # then we trunct he dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)
        else:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __truncate_channel__(self, index):
        #self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.cifar_dataobj.data[gs_index, :, :, 1] = self.cifar_dataobj.data[gs_index, :, :, 0]
            self.cifar_dataobj.data[gs_index, :, :, 2] = self.cifar_dataobj.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10ColorGrayScaleOverSampled(data.Dataset):
    '''
    Here we conduct oversampling strategy (over the underrepresented domain) in mitigating the data bias
    '''
    def __init__(self, root, dataidxs=None, gray_scale_indices=None,
                    train=True, transform_color=None, transofrm_gray_scale=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_color = transform_color
        self.transofrm_gray_scale = transofrm_gray_scale
        self.target_transform = target_transform
        self._gray_scale_indices = gray_scale_indices
        self.download = download

        self.cifar_dataobj = CIFAR10(self.root, self.train, None, self.target_transform, self.download)

        # we need to trunc the channle first
        self.__truncate_channel__(index=gray_scale_indices)
        # then we trunct he dataset
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)
        else:
            data = self.cifar_dataobj.data
            target = np.array(self.cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __truncate_channel__(self, index):
        #self._gray_scale_indices = index
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.cifar_dataobj.data[gs_index, :, :, 1] = self.cifar_dataobj.data[gs_index, :, :, 0]
            self.cifar_dataobj.data[gs_index, :, :, 2] = self.cifar_dataobj.data[gs_index, :, :, 0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        #if self.transform is not None:
        if index in self._gray_scale_indices:
            if self.transofrm_gray_scale is not None:
                img = self.transofrm_gray_scale(img)
        else:
            if self.transform_color is not None:
                img = self.transform_color(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, dataidxs=None, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolderTruncated, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.dataidxs = dataidxs

        ### we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            #self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels