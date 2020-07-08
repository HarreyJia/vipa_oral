import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import os
import cv2
import random
from torch.utils.data.dataset import Dataset
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_EPOCHS = 50
NUM_ROUTING_ITERATIONS = 3
RAND_NUM = random.randint(0,100)

ROOT_PATH = "/disk2/jlx/oral/teeth_preprocess/output/"
HEALTHY_PATH = os.path.join(ROOT_PATH, "healthy/cropped_resized/")
UNHEALTHY_PATH = os.path.join(ROOT_PATH, "unhealthy_augmentation/cropped/")
OUTPUT_PATH = "/disk2/jlx/oral/binary_classification/output/"

def chunks(arr, m):
    nchunk = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + nchunk] for i in range(0, len(arr), nchunk)]


def folder_valiation(arr, number):
    training_set = []
    test_set = []
    for j in range(len(arr)):
        if number == j:
            test_set.extend(arr[j])
        else:
            training_set.extend(arr[j])
    return training_set, test_set


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class MyDataset(Dataset):
    def __init__(self, images,labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        img = self.images[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=3)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 38 * 38, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256 * 256),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":

    oral_dataset = []
    oral_label = []
    # Read HEALTHY data
    for image_instance in os.listdir(HEALTHY_PATH):
        healthy_image = cv2.imread(HEALTHY_PATH + image_instance, 0)
        healthy_image = np.expand_dims(healthy_image, 0)
        oral_dataset.append(healthy_image)
        oral_label.append(0)

    # Read UNHEALTHY data
    for image_instance in os.listdir(UNHEALTHY_PATH):
        unhealthy_image = cv2.imread(UNHEALTHY_PATH + image_instance, 0)
        unhealthy_image = np.expand_dims(unhealthy_image, 0)
        oral_dataset.append(unhealthy_image)
        oral_label.append(1)

    # Shuffle the dataset
    random.seed(RAND_NUM)
    random.shuffle(oral_dataset)
    random.seed(RAND_NUM)
    random.shuffle(oral_label)

    # Clip the training set and test set
    dataset_chunks = chunks(oral_dataset, 10)
    label_chunks = chunks(oral_label, 10)
    training_set, test_set = folder_valiation(dataset_chunks, 9)
    training_label, test_label = folder_valiation(label_chunks, 9)

    # Prepare the data loader
    train_loader = torch.utils.data.DataLoader(MyDataset(training_set, training_label), batch_size=BATCH_SIZE, drop_last=True)
    test_loader = torch.utils.data.DataLoader(MyDataset(test_set, test_label), batch_size=BATCH_SIZE, drop_last=True)

    model = CapsuleNet()
    model.cuda()
    optimizer = Adam(model.parameters())
    capsule_loss = CapsuleLoss()

    train_loss_curve = []
    test_loss_curve = []
    train_acc_curve = []
    test_acc_curve = []

    # train(train_loader, test_loader)
    print("------------Process STARTS--------------")
    for epoch in range(NUM_EPOCHS):
        # TRAIN MODE
        print("The %d EPOCH starts:" % (epoch + 1))
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_id, (data, target) in enumerate(train_loader, 0):
            target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
            data, target = Variable(data.float()), Variable(target)
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            classes, reconstructions = model(data, target)
            loss = capsule_loss(data, target, classes, reconstructions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

            train_acc += sum(np.argmax(y.data.cpu().numpy(), 1) == 
                                np.argmax(target.data.cpu().numpy(), 1)) / float(BATCH_SIZE)
            
        print("train accuracy: %.4f, train loss: %.4f" % (train_acc / len(train_loader), train_loss / len(train_loader)))
        train_acc_curve.append(train_acc / len(train_loader))
        train_loss_curve.append(train_loss / len(train_loader))
            
        model.eval()
        test_loss = 0
        test_acc = 0
        for batch_id, (data, target) in enumerate(test_loader):
            target = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
            data, target = Variable(data.float()), Variable(target)
            data, target = data.cuda(), target.cuda()

            classes, reconstructions = model(data)
            loss = capsule_loss(data, target, classes, reconstructions)

            test_loss += loss.item()
        
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

            test_acc += sum(np.argmax(y.data.cpu().numpy(), 1) == 
                                np.argmax(target.data.cpu().numpy(), 1)) / float(BATCH_SIZE)

        print("test accuracy: %.4f, test loss: %.4f" % (test_acc / len(test_loader), test_loss / len(test_loader)))
        test_acc_curve.append(test_acc / len(test_loader))
        test_loss_curve.append(test_loss / len(test_loader))

        if (epoch % 10 == 0 and epoch != 0) or epoch == NUM_EPOCHS - 1:
            
            params_output_path = os.path.join(OUTPUT_PATH, "model_params/")
            accuracy_output_path = os.path.join(OUTPUT_PATH, "accuracy/")
            torch.save(model.state_dict(), params_output_path + 'cropped_params_' + str(epoch) + '.pth')  

            plt.figure(figsize=(10,5))
            plt.plot(train_acc_curve, label="Train")
            plt.plot(test_acc_curve, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.savefig(accuracy_output_path + "cropped_acc.png")

            plt.figure(figsize=(10,5))
            plt.plot(train_loss_curve, label="Train")
            plt.plot(test_loss_curve, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(accuracy_output_path + "cropped_loss.png")
    
    print("------------Process ENDS--------------")