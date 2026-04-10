import cv2
import numpy as np
import os

image_dir = "data/images"
mask_dir = "data/masks"

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    h, w, _ = img.shape

    # Create fake mask (white square in center)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//4:3*h//4, w//4:3*w//4] = 255

    cv2.imwrite(os.path.join(mask_dir, img_name), mask)

print("Dummy masks created!")