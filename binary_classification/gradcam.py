import argparse
import cv2
import numpy as np
from torch.autograd import Variable
import torch
from torch.autograd import Function
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

NUM_CLASSES = 5

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # for name, module in self.model.module.named_children():
        x = self.model(x)
        # if name in self.target_layers:
        x.register_hook(self.save_gradient)
        outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model.module.named_children():
            if name == 'conv1':
                x = module(x)
                # print(name, x.shape)
            elif name == 'primary_capsules':
                target_activations, x = self.feature_extractor(x)
                # print(target_activations)
                # print(name, x.shape)
            elif name == 'digit_capsules':
                x = module(x)
                # print(name, x.shape)
        
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
            output = output.squeeze().transpose(0, 1)
            print(output.shape)
        else:
            features, output = self.extractor(input)

        classes = (output ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        _, max_length_indices = classes.max(dim=1) # (batch_size)
        y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        if index == None:
            index = np.argmax(y.cpu().data.numpy(), 1)
        print(index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        print(one_hot)
        one_hot[0][index] = 1
        print(one_hot)
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        print("grad shape is: ", grads_val.shape)

        target = features[-1]
        # target = target.cpu().data.numpy()[0, :]
        target = target.cpu().data.numpy()
        print("target: ", target.shape)

        # weights = np.mean(grads_val, axis=(1, 2))
        weights = grads_val
        print("weight: ", weights.shape)
        cams = np.zeros(target.shape, dtype=np.float32)

        # for i, w in enumerate(weights):
        #     cams += w * target[i, :, :]
        cams = weights * target
        
        print("cam:", cams.shape)

        features_heat_maps = []
        for i in range(cams.shape[0]):
            img = input[i].detach().cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)
            # cam = np.maximum(cam, 0)
            cam = cv2.resize(cams[i], input.shape[2:])
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            # heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img.transpose((1, 2, 0)) * 255)
            # cam = cam / np.max(cam)
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            features_heat_maps.append(
                transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        features_heat_maps = torch.stack(features_heat_maps)
        return features_heat_maps
