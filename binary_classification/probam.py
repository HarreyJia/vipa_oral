import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from resnet import resnet
from torch import nn


class ProbAM:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, images):
        image_size = (images.size(-1), images.size(-2))

        for name, module in self.model.named_children():
            if name == 'conv1':
                out = module(images)
                conv1_heat_maps = out.mean(dim=1, keepdim=True)
            elif name == 'primary_capsules':
                ### 1.resnet
                # features = resnet()
                # out = features(out)
                # features = out
                out = module(out)
            elif name == 'digit_capsules':
                # fc = nn.Linear(out.size(-1), 46208 * 8).cuda()
                # out = fc(out)
                # out = out.view(out.size(0), 46208, 8)

                # out, probs = module(out)
                # classes = out.norm(dim=-1)
                # prob = (probs * classes.unsqueeze(dim=-1)).sum(dim=-1).permute(1, 3, 0, 2).sum(dim=1)
                # prob = prob.view(prob.size(0), -1, 5)
                # print(prob.shape)

                out = out.view(out.size(0), 38, 38, 8, 32)
                prob = out[:,:,:,0,:].sum(dim=-1)

                print(prob.shape)

                # prob = prob.view(prob.size(0), *features.size()[-2:], -1)
                # prob = prob.permute(0, 3, 1, 2).sum(dim=1)

                features_heat_maps = []
                f_heat_maps = []
                for i in range(prob.size(0)):
                    img = images[i].detach().cpu().numpy()
                    img = img - np.min(img)
                    if np.max(img) != 0:
                        img = img / np.max(img)
                    mask = cv2.resize(prob[i].detach().cpu().numpy(), image_size)

                    temp = np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
                    temp = temp - np.min(temp)
                    if np.max(temp) != 0:
                        temp = temp / np.max(temp)
                    # single_heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * temp), cv2.COLORMAP_JET))
                    f_heat_maps.append(
                        transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * temp), cv2.COLOR_BGR2RGB)))

                    mask = mask - np.min(mask)
                    if np.max(mask) != 0:
                        mask = mask / np.max(mask)
                    heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
                    cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
                    cam = cam - np.min(cam)
                    if np.max(cam) != 0:
                        cam = cam / np.max(cam)
                    features_heat_maps.append(
                        transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
                features_heat_maps = torch.stack(features_heat_maps)
                f_heat_maps = torch.stack(f_heat_maps)
        return conv1_heat_maps, features_heat_maps, f_heat_maps
