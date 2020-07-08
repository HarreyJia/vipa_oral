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
NUM_CLASSES = 2
NUM_EPOCHS = 20

RAND_NUM = random.randint(0,100)

ROOT_PATH = "/disk2/jlx/oral/"
DATA_PATH = os.path.join(ROOT_PATH, "teeth_preprocess/output/")
HEALTHY_PATH = os.path.join(DATA_PATH, "healthy/")
UNHEALTHY_PATH = os.path.join(DATA_PATH, "unhealthy/") 
OUTPUT_PATH = os.path.join(ROOT_PATH, "binary_classification/output/")

os.environ["CUDA_VISIBLE_DEVICES"]="0,3"

def chunks(arr, m):
    nchunk = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + nchunk] for i in range(0, len(arr), nchunk)]


def folder_validation(arr, number):
    training_set = []
    test_set = []
    for j in range(len(arr)):
        if number == j:
            test_set.extend(arr[j])
        else:
            training_set.extend(arr[j])
    return training_set, test_set


class MyDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.tranforms = transforms

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


def read_data(path):
    dataset = []
    for image_instance in os.listdir(path):
        image = cv2.imread(path + image_instance, 0)
        image = np.expand_dims(image, 0)
        dataset.append(image)
    return dataset


def divide_dataset(dataset, chunks_num):
    certain_chunks = chunks(dataset, chunks_num)
    certain_training_set, certain_test_set = folder_validation(certain_chunks, chunks_num - 1)
    return certain_training_set, certain_test_set


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='origin',
                        help='Use the origin dataset or the cropped dataset')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    TYPE_PATH = args.type + "_resized/"

    # Read HEALTHY data
    healthy_dataset = read_data(HEALTHY_PATH + TYPE_PATH)
    training_healthy_set, test_healthy_set = divide_dataset(healthy_dataset, 5)
    
    # Read UNHEALTHY RAW data
    unhealthy_dataset = read_data(UNHEALTHY_PATH + TYPE_PATH)
    training_unhealthy_set, test_unhealthy_set = divide_dataset(unhealthy_dataset, 5)

    print("All the Data from " + TYPE_PATH + " has been read.")

    # Merge the healthy data and the unhealthy data(after augmentation) into the training set
    training_set = []
    training_label = []
    training_set.extend(training_healthy_set)
    training_set.extend(training_unhealthy_set)
    training_label.extend([0] * len(training_healthy_set))
    training_label.extend([1] * len(training_unhealthy_set))

    print("Training set is ready and the shape is %d." % len(training_set))

    # Merge the healthy data and the unhealthy data(without augmentation) into the test set
    test_set = []
    test_label = []
    test_set.extend(test_healthy_set)
    test_set.extend(test_unhealthy_set)
    test_label.extend([0] * len(test_healthy_set))
    test_label.extend([1] * len(test_unhealthy_set))

    print("Test set is ready and the shape is %d." % len(test_set))

    # Prepare the data loader
    train_loader = torch.utils.data.DataLoader(MyDataset(training_set, training_label), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(MyDataset(test_set, test_label), batch_size=BATCH_SIZE)

    model = CapsuleNet()
    model = nn.DataParallel(model).cuda()
    optimizer = Adam(model.parameters())
    capsule_loss = CapsuleLoss()

    train_loss_curve = []
    test_loss_curve = []
    train_acc_curve = []
    test_acc_curve = []

    classes = [0,1]
    weight = compute_class_weight('balanced', classes, training_label)

    # train(train_loader, test_loader)
    for epoch in range(NUM_EPOCHS):
        # TRAIN MODE
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_id, (data, target) in tqdm(enumerate(train_loader, 0), desc='Batch', total=len(train_loader)):
            target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
            weighted_target = [weight] * BATCH_SIZE
            weighted_target = Variable(torch.tensor(weighted_target).float() * target).cuda()
            data, target = Variable(data.float()), Variable(target)
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            classes, reconstructions = model(data, target)
            loss = capsule_loss(data, weighted_target, classes, reconstructions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

            cur_train_acc = sum(np.argmax(y.data.cpu().numpy(), 1) == 
                                np.argmax(target.data.cpu().numpy(), 1)) / float(BATCH_SIZE)
            train_acc += cur_train_acc
            # print(classes[:, 0])
            print(loss.item())
            print(cur_train_acc)
        print("[%d/%d] Train accuracy: %.4f, Train loss: %.4f" % (epoch + 1, NUM_EPOCHS,train_acc / len(train_loader), train_loss / len(train_loader)))
        train_acc_curve.append(train_acc / len(train_loader))
        train_loss_curve.append(train_loss / len(train_loader))
            
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(test_loader):
                target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
                data, target = Variable(data.float()), Variable(target)
                data, target = data.cuda(), target.cuda()
                classes, reconstructions = model(data)
                loss = capsule_loss(data, target, classes, reconstructions)

                test_loss += loss.item()
            
                _, max_length_indices = classes.max(dim=1) # (batch_size)
                y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data) # (batch_size, num_classes)
                # pred_labels.extend(pred.cpu().numpy())
                test_acc += sum(np.argmax(y.data.cpu().numpy(), 1) == 
                                    np.argmax(target.data.cpu().numpy(), 1))
            # test_acc = accuracy_score(target_labels, pred_labels)
            print("[%d/%d] Test accuracy: %.4f, Test loss: %.4f" % (epoch + 1, NUM_EPOCHS, test_acc / len(test_set), test_loss / len(test_loader)))
            test_acc_curve.append(test_acc / len(test_set))
            test_loss_curve.append(test_loss / len(test_loader))

        if (epoch + 1) % 5 == 0:
            
            params_output_path = os.path.join(OUTPUT_PATH + TYPE_PATH, "model_params/")
            accuracy_output_path = os.path.join(OUTPUT_PATH + TYPE_PATH, "accuracy/")
            torch.save(model.state_dict(), params_output_path + 'params_' + str(epoch + 1) + '.pth')  

            plt.figure(figsize=(5,5))
            plt.plot(train_acc_curve, label="Train")
            plt.plot(test_acc_curve, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(accuracy_output_path + "accuracy.png")

            plt.figure(figsize=(5,5))
            plt.plot(train_loss_curve, label="Train")
            plt.plot(test_loss_curve, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(accuracy_output_path + "loss.png")

