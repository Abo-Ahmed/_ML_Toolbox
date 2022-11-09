from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

print("step 1")
model = VGG16(classes= 3 , weights= None)
print(model.summary())

print("step 2")
img_path = '/content/dog.jpg'
z = image.load_img(img_path,color_mode='rgb', target_size=(224, 224))
print("z 0",z)

print("step 3")
z = image.img_to_array([z])
print("z 1" , z.shape,z)

z = np.expand_dims(z, axis=0)
print("z 3", z.shape,z)

print("step 4")
z = preprocess_input(z)
print("z 4",z)
features = model.predict(z)
print("features",features)
p = decode_predictions(features)
print("p", p)