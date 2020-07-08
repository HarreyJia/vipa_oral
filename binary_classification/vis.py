import torch
from torch.optim import Adam
import os
import cv2
import random
from torch.utils.data.dataset import Dataset
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch import nn
from torch.autograd import Variable
from CapsNet import CapsuleLoss, CapsuleNet
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from probam import ProbAM
from gradcam import *

BATCH_SIZE = 32
NUM_CLASSES = 5

dict = {
  "radicular_cyst": 1,
  "dentigerous_cyst": 2,
  "keratocyst": 3,
  "ameloblastoma": 4
}

ROOT_PATH = "/disk2/jlx/oral/"
DATA_PATH = os.path.join(ROOT_PATH, "teeth_preprocess/output/")
HEALTHY_PATH = os.path.join(DATA_PATH, "healthy/cropped_resized/")
UNHEALTHY_PATH = os.path.join(DATA_PATH, "multi_unhealthy/") 

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

class MyDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


def read_data(path):
    dataset = []
    for image_instance in os.listdir(path):
        image = cv2.imread(path + image_instance, 0)
        image = np.expand_dims(image, -1)
        dataset.append(image)
    return dataset


if __name__ == "__main__":
    
    # Read UNHEALTHY RAW data
    unhealthy_dataset = []
    unhealthy_label = []

    # Read UNHEALTHY data
    for certain_unhealthy_class in os.listdir(UNHEALTHY_PATH):
        unhealthy_class_label = dict[certain_unhealthy_class]
        certain_unhealthy_path = os.path.join(UNHEALTHY_PATH, certain_unhealthy_class + "/")
        certain_unhealthy_dataset = read_data(certain_unhealthy_path)
        print("The %s data has been loaded and the size is %d." % (certain_unhealthy_class, len(certain_unhealthy_dataset)))
        
        unhealthy_dataset.extend(certain_unhealthy_dataset)
        unhealthy_label.extend([unhealthy_class_label] * len(certain_unhealthy_dataset))
    
    unhealthy_loader = torch.utils.data.DataLoader(MyDataset(unhealthy_dataset, unhealthy_label), batch_size=BATCH_SIZE)

    model = CapsuleNet()
    AM_method = ProbAM(model)

    model = nn.DataParallel(model).cuda()
    model_params = torch.load("/disk2/jlx/oral/binary_classification/output/multi-classification/model_params/params_20.pth")
    model.load_state_dict(model_params)

    # feature_module = None
    # target_layer = None
    # for name, module in model.module.named_children():
    #     if name == 'primary_capsules':
    #         feature_module = module
    #     elif name == 'digit_capsules':
    #         target_layer = module

    nrow = 8
    for batch_id, (data, target) in tqdm(enumerate(unhealthy_loader), desc='Batch', total=len(unhealthy_loader)):
        data= Variable(data.float()).cuda()
        print(target)

        # grad_cam = GradCam(model=model, feature_module=feature_module, \
        #                 target_layer_names=target_layer, use_cuda=True)

        # # If None, returns the map for the highest scoring category.
        # # Otherwise, targets the requested index.
        # target_index = None
        # features_heat_maps = grad_cam(data, target_index)

        conv1_heat_maps, features_heat_maps, f_heat_maps = AM_method(data)

        save_image(conv1_heat_maps, filename='./output/vis/vis_batch%d_conv1.png' % batch_id, nrow=nrow, normalize=True, padding=4, pad_value=255)
        save_image(features_heat_maps[16], filename='./output/vis/vis_batch%d_features.png' % batch_id, nrow=2, padding=4, pad_value=255)
        save_image(data[16], filename='./output/vis/vis_batch%d_f.png' % batch_id, nrow=2, padding=4, pad_value=255)
