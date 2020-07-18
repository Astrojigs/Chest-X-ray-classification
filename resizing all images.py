import os
from PIL import Image
images_path = 'Covid19-dataset/test/Covid'
size = 540, 583
# for image_name, num in zip(os.listdir(images_path), range(687)):
#     full_path = os.path.join(images_path, image_name)
#     # changing resolution
#     img = Image.open(full_path)
#     resized_img = img.resize(size, Image.ANTIALIAS)
#     resized_img.save(str(num) + '.jpg')
