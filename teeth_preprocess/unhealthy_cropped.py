import os
import cv2
import matplotlib.pyplot as plt
import copy

INPUT_PATH = "/home/disk1/jlx/oral/data/unhealthy/"
OUTPUT_PATH = "./output/unhealthy/"
count = 0

for instance in os.listdir(INPUT_PATH):
    count += 1
    print("Loading image number %d" % count)
    img_address = os.path.join(INPUT_PATH, instance)
    img = cv2.imread(img_address, 0)

    # OUTPUT origin image(resized to 256*256)
    resize_size = (256, 256)
    img_resized = cv2.resize(img, resize_size, interpolation = cv2.INTER_CUBIC)
    origin_resized_img_path = os.path.join(OUTPUT_PATH, "origin_resized/" + instance)
    cv2.imwrite(origin_resized_img_path, img_resized)

    img_copy = copy.deepcopy(x=img)
    img_shape = img_copy.shape
    img_cropped = img_copy[int(img_shape[0] / 6):, int(img_shape[1] / 7):int(6 * img_shape[1] / 7)]

    # plt.imshow(X=img_cropped, cmap='gray')
    # plt.show() 

    # OUTPUT cropped image(resized to 256*256)
    resize_size = (256, 256)
    img_resized = cv2.resize(img_cropped, resize_size, interpolation = cv2.INTER_CUBIC)
    origin_resized_img_path = os.path.join(OUTPUT_PATH, "cropped_resized/" + instance)
    cv2.imwrite(origin_resized_img_path, img_resized)

