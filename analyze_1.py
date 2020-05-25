from numpy import loadtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import cv2
from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')
#dataset = loadtxt('sign_mnist_train.csv', delimiter=',')
#dataset2 = loadtxt('sign_mnist_test.csv', delimiter=',')
labels = train['label'].values
unique_val = np.array(labels)
train.drop('label',axis=1,inplace=True)
images = train.values
images = np.array([np.reshape(i,(28,28)) for i in images])
images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
x_train = x_train / 255
x_test = x_test / 255
print(x_train.shape[0])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
plt.imshow(images[0].reshape(28,28))
plt.show()
plt.imshow(images[1].reshape(28,28))
plt.show()
print(x_train[0].shape)

model = load_model('CNNmodel.h5')
print(model.predict(x_train[0:2]))
pred_probab = model.predict(x_train[0:2])
pred_class = list(pred_probab[1]).index(max(pred_probab[1]))
print(pred_class)
