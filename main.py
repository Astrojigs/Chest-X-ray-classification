# Classification of cases from Chest-xray images
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

data_dir = 'Covid19-dataset'

# Creating paths
train_path = 'Covid19-dataset/train'
test_path = 'Covid19-dataset/test'

# number of images in training folder
print(len(os.listdir(os.path.join(train_path, 'Covid'))))
# number of images for Normal, Covid, Viral Pneumonia = 70,111,70

# Dimensionality study
dim1 = []
dim2 = []
test_covid_path = test_path + '/Covid'

for image_name in os.listdir(os.path.join(test_path, 'Covid')):
    img = imread(os.path.join(test_covid_path, image_name))
    # print(img.shape)
    '''debug for tuple length 2'''
    # if len(img.shape)==2:
    #     print(image_name)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
# print(np.mean(dim1), np.mean(dim2)) = 728.2 782.6

# Keeping dimensions of images same
image_shape = (540, 583, 3)

# NOTE: Add Rescaling if you are not generating images
img_gen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                             rescale=1 / 255, zoom_range=0.3)

# img_gen.flow_from_directory(test_path)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Settings
batch_size = 22

train_image_gen = img_gen.flow_from_directory(train_path, target_size=image_shape[:2],
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical')
test_image_gen = img_gen.flow_from_directory(test_path, target_size=image_shape[:2],
                                             color_mode='rgb',
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)
model.summary()

# Result index
# print(train_image_gen.class_indices) ={'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}


# results = model.fit(train_image_gen, epochs=30, validation_data=test_image_gen,
#                     callbacks=[early_stop])

# saving the model
# model.save('covid_pred.h5', overwrite=True)

model = tf.keras.models.load_model('covid_pred.h5')
pred = model.predict_classes(test_image_gen)
# print(pred)
'''[0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 2 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 2 2 1 2 2 1 1 2 2 1 1 2 2 2 2 2 2]'''

# print(f"test image gen: {test_image_gen.classes}")
'''test image gen: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]'''

print(f"classification report:\n {classification_report(test_image_gen.classes, pred)}")
