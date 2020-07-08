import os
import numpy as np
import cv2
import copy

'''
    This part is used to realize data augmentation, which probably contains:
    (1) FLIP
    (2) ROTATE
    (3) SAULT NOISE
    (4) BRIGHTER & DARKE
    (5) TRANSLATION(LEFT & RIGHT)
'''

def Flip(image, flip_type):
    return cv2.flip(image, flip_type)


def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image


def SaltAndPepper(src,percetage):  
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
    for i in range(SP_NoiseNum): 
        randR=np.random.randint(0,src.shape[0]-1) 
        randG=np.random.randint(0,src.shape[1]-1)
        if np.random.randint(0,1)==0: 
            SP_NoiseImg[randR,randG]=0 
        else: 
            SP_NoiseImg[randR,randG]=255 
    return SP_NoiseImg


def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0,w):
        for xj in range(0,h):
            if len(image_copy.shape) == 2:
                image_copy[xj,xi] = int(image[xj,xi]*percetage)
            else:
                image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
                image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
                image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy


def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0,w):
        for xj in range(0,h):
            if len(image_copy.shape) == 2:
                image_copy[xj,xi] = np.clip(int(image[xj,xi]*percetage),a_max=255,a_min=0)
            else:
                image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
                image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
                image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy


def translation(image, direction):
    image_copy = image.copy()
    rows,cols = image_copy.shape
    matrix = [1,0,direction]
    M = np.float32([matrix, [0,1,0]])
    dst = cv2.warpAffine(image_copy, M, (cols, rows))
    return dst


def image_augmentation(img):

    img_aug = []
    img_aug.append(img)

    img_origin = img[:,:,0]

    img_flip = Flip(img_origin ,1)
    img_flip = np.expand_dims(img_flip, -1)
    img_aug.append(img_flip)

    img_sault = SaltAndPepper(img_origin, 0.05)
    img_sault = np.expand_dims(img_sault, -1)
    img_aug.append(img_sault)

    img_darker = darker(img_origin)
    img_darker = np.expand_dims(img_darker, -1)
    img_aug.append(img_darker)

    img_brighter = brighter(img_origin)
    img_brighter = np.expand_dims(img_brighter, -1)
    img_aug.append(img_brighter)

    img_right_translation = translation(img_origin, 10)
    img_right_translation = np.expand_dims(img_right_translation, -1)
    img_aug.append(img_right_translation)

    img_left_translation = translation(img_origin, -10)
    img_left_translation = np.expand_dims(img_left_translation, -1)
    img_aug.append(img_left_translation)

    return img_aug