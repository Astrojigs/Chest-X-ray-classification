from main import img_gen
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
# batch_size: At a time, how many image should be created.
count = 0

# img = load_img('Covid19-dataset/train/Normal/30.jpg')

origin_and_des_path = 'Covid19-dataset/train/Normal'
imgs = []
for img_name in os.listdir(origin_and_des_path):
    image = load_img(os.path.join(origin_and_des_path, img_name))
    image_arr = img_to_array(image)
    imgs.append(image_arr)
imgs_arr = np.array(imgs)
print(f"images_array look like: {imgs_arr.shape}")

# img_arr = img_to_array(img)
# X = img_arr.reshape((1,) + img_arr.shape)
# print(X.shape)

for batch in img_gen.flow(imgs_arr, batch_size=10, save_to_dir=origin_and_des_path, save_prefix='addon',
                          save_format='jpg'):
    count += 1
    if count > 40:
        break
# 40xbatch_size = number of imges generated
