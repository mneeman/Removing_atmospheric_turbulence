from os import listdir
from os.path import join, isdir
import random



from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize,  CenterCrop, Normalize, ToTensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def train_transform(image_size):
    return Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

def test_transform(image_size):
    return Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])




class MyDataset(Dataset):
    def __init__(self, opt):

        self.data_dir = opt.dataset_dir
        self.reference_dataset_path = opt.reference_dataset_path
        self.transform = train_transform(opt.img_size)
        self.filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if isdir(join(self.data_dir, x))]
        self.sample_num = opt.sample_num
        self.windows = opt.windows

    def __getitem__(self, index):

        folder = join(self.data_dir, self.filenames[index])
        image_names_path = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        rand_imgs_path = random.sample(image_names_path ,k=self.sample_num)
        rand_imgs = [self.transform(Image.open(join(folder, rand_imgs_path[x]))) for x in range(len(rand_imgs_path))]
        
        # reference image before adding the turbulence
        if self.windows:
            ref_image_name = folder.split('\\')[-1].split('_')
        else:
            ref_image_name = folder.split('/')[-1].split('_')
        
        if ref_image_name[0] == 'inside':
            ref_image_path = join(self.reference_dataset_path, 'inside_city', ref_image_name[-1] + '.jpg')
        else: 
            ref_image_path = join(self.reference_dataset_path, ref_image_name[0], ref_image_name[1] + '.jpg')
        ref_image = Image.open(ref_image_path)

        rand_imgs = torch.stack(rand_imgs)
        ref_image = self.transform(ref_image)
        return rand_imgs, ref_image

    def __len__(self):
        return len(self.filenames)
    

class MyDataset_test(Dataset):
    def __init__(self, opt):

        self.data_dir = opt.test_dataset_path
        self.transform = test_transform(opt.img_size)
        self.filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if isdir(join(self.data_dir, x))]
        self.sample_num = opt.sample_num

    def __getitem__(self, index):

        folder = join(self.data_dir, self.filenames[index])
        image_names_path = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        rand_imgs_path = random.sample(image_names_path ,k=self.sample_num)
        rand_imgs = [self.transform(Image.open(join(folder, rand_imgs_path[x]))) for x in range(len(rand_imgs_path))]
        rand_imgs = torch.stack(rand_imgs)

        return rand_imgs, self.filenames[index]

    def __len__(self):
        return len(self.filenames)

class MyDataset_test_moving(Dataset):
    def __init__(self, opt):

        self.data_dir = opt.test_moving_path
        self.transform = test_transform(opt.img_size)
        self.filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if isdir(join(self.data_dir, x))]
        self.sample_num = opt.sample_num

    def __getitem__(self, index):

        folder = join(self.data_dir, self.filenames[index])
        image_names_path = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        imgs_moving = []
        i = 0
        while i+self.sample_num < len(image_names_path):
            imgs_path = image_names_path[i:i+self.sample_num]
            imgs = [self.transform(Image.open(join(folder, imgs_path[x]))) for x in range(len(imgs_path))]
            imgs = torch.stack(imgs)
            imgs_moving.append(imgs)
            i += self.sample_num

        return imgs_moving, self.filenames[index]

    def __len__(self):
        return len(self.filenames)