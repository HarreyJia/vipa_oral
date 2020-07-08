import os
import cv2
import matplotlib.pyplot as plt
import copy

INPUT_PATH = "/disk2/jlx/oral/data/unhealthy_classification_dataset/"
OUTPUT_PATH = "/disk2/jlx/oral/teeth_preprocess/output/multi_unhealthy/"

dict = {
  "根尖囊肿": "radicular_cyst",
  "含牙囊肿": "dentigerous_cyst",
  "角化囊肿": "keratocyst",
  "成釉细胞瘤": "ameloblastoma"
}

for certain_unhealthy_class in os.listdir(INPUT_PATH):
    if certain_unhealthy_class == "问题":
        continue
    print("Loading image from %s." % certain_unhealthy_class)
    certain_unhealthy_path = os.path.join(INPUT_PATH, certain_unhealthy_class)

    for img_instance in os.listdir(certain_unhealthy_path):
        img_address = os.path.join(certain_unhealthy_path, img_instance)
        img = cv2.imread(img_address, 0)

        img_copy = copy.deepcopy(x=img)
        img_shape = img_copy.shape
        img_cropped = img_copy[int(img_shape[0] / 6):, int(img_shape[1] / 7):int(6 * img_shape[1] / 7)]

        # OUTPUT cropped image(resized to 256*256)
        resize_size = (256, 256)
        img_resized = cv2.resize(img_cropped, resize_size, interpolation = cv2.INTER_CUBIC)
        certain_unhealthy = dict[certain_unhealthy_class]
        resized_img_path = os.path.join(OUTPUT_PATH, certain_unhealthy + "/" + img_instance)
        print(resized_img_path)
        cv2.imwrite(resized_img_path, img_resized)
