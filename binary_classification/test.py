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
from augmentation import image_augmentation
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='origin',
                        help='Use the origin dataset or the cropped dataset')
    args = parser.parse_args()
    return args


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
    model = nn.DataParallel(model).cuda()
    model_params = torch.load("/disk2/jlx/oral/binary_classification/output/multi-classification/model_params/params_20.pth")
    model.load_state_dict(model_params)
    capsule_loss = CapsuleLoss()

    unhealthy_loss = 0
    unhealthy_acc = 0

    for batch_id, (data, target) in tqdm(enumerate(unhealthy_loader), desc='Batch', total=len(unhealthy_loader)):
        target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
        data, target = Variable(data.float()), Variable(target)
        data, target = data.cuda(), target.cuda()
        classes, reconstructions = model(data)
        loss = capsule_loss(data, target, classes, reconstructions)

        unhealthy_loss += loss.item()
    
        _, max_length_indices = classes.max(dim=1) # (batch_size)
        y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data) # (batch_size, num_classes)

        unhealthy_acc += sum(np.argmax(y.data.cpu().numpy(), 1) == 
                            np.argmax(target.data.cpu().numpy(), 1))

    print("Test accuracy: %.4f, Test loss: %.4f" % (unhealthy_acc / len(unhealthy_dataset), unhealthy_loss / len(unhealthy_loader)))
