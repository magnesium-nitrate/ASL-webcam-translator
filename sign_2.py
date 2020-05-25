#https://www.kaggle.com/datamunge/sign-language-mnist
from numpy import loadtxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
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

batch_size = 128
num_classes = 24
epochs = 50
#X = dataset[:,1:785]
#Y = dataset[:,0]
#X2 = dataset[:,1:785]
#Y2 = dataset[:,0]
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)

