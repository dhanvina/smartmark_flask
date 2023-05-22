import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , Activation, Flatten , Conv2D, MaxPooling2D
import cv2
import pickle
from keras.models import load_model
   

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test , y_test) = mnist.load_data()
x_train.shape

plt.imshow(x_train[5])
plt.show()
plt.imshow(x_train[5] , cmap = plt.cm.binary)

plt.imshow(x_train[4])
plt.show()
plt.imshow(x_train[4] , cmap = plt.cm.binary)

plt.imshow(x_train[2])
plt.show()
plt.imshow(x_train[2] , cmap = plt.cm.binary)

plt.imshow(x_train[3])
plt.show()
plt.imshow(x_train[3] , cmap = plt.cm.binary)

plt.imshow(x_train[15])
plt.show()
plt.imshow(x_train[15] , cmap = plt.cm.binary)

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)
plt.imshow(x_train[0] , cmap = plt.cm.binary)

img_size = 28

x_trainer = np.array(x_train).reshape(-1,img_size,img_size,1)
x_tester = np.array(x_test).reshape(-1,img_size,img_size,1)
print('Training shape' , x_trainer.shape)
print('Testing shape' , x_tester.shape)

model = Sequential()

model.add(Conv2D(32 , (3,3) , activation = 'relu' , input_shape= x_trainer.shape[1:]))
# model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# model.add(Conv2D(64 , (3,3) , activation = 'relu'))
# model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# compile model that we have created for handwritten digit recognition project
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

# fit x_trainer , y_train to the model to see accuracy of model:
model.fit(x_trainer,y_train, epochs = 10 , validation_split = 0.3 , batch_size = 128,verbose=1)

test_loss, test_acc = model.evaluate(x_tester, y_test)
print('Test loss on 10,000 test samples' , test_loss)
print('Validation Accuracy on 10,000 samples' , test_acc)

test_loss, test_acc = model.evaluate(x_tester, y_test)
print('Test loss on 10,000 test samples' , test_loss)
print('Validation Accuracy on 10,000 samples' , test_acc)

pickle.dump(model,open('digitRec.pkl','wb'))
# load_model = pickle.load(open('digitRec.pkl','rb'))

model.save('model.h5')
   
