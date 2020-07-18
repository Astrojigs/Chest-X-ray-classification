from main import imread, np, tf, os, image_shape, plt, train_path, test_path
# Note: While generating new images do not rescale/normalize(/255) them, remove rescale(normalize) option from img_gen
# {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}
# prediction of single image

train_or_test = [train_path, test_path]
sub_dir = ['Covid', 'Normal', 'Viral Pneumonia']
random_img_path = os.path.join(train_or_test[np.random.randint(0, 1)], sub_dir[np.random.randint(0, 2)])
random_list_imgs = os.listdir(random_img_path)


single_img = imread(os.path.join(random_img_path, random_list_imgs[np.random.randint(0, len(random_list_imgs))]))

# normalising
single_img = single_img / 255
single_reshaped_img = np.reshape(single_img, (1, 540, 583, 3))

# making prediction
model = tf.keras.models.load_model('covid_pred.h5')
prediction = model.predict_classes(single_reshaped_img)
print(prediction)

# displaying this image
plt.imshow(single_img)
plt.xlabel(f"The predicted case:{sub_dir[int(prediction)]}")
plt.title(f"The actual: {random_img_path}")
plt.show()
