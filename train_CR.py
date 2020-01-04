import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import load_model

seed = 42
np.random.seed(seed)

data = pd.read_csv('data.csv')

X = []
Y = data['y']
del data['y']
del data['Character']
for i in range(data.shape[0]):
    flat_pixels = data.iloc[i].values[1:]
    image = np.reshape(flat_pixels, (28,28))
    X.append(image)
#print(X[0])
X = np.array(X)
Y = np.array(Y)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.30, random_state=seed)

Y_test_for_accuracy_matrix = Y_test.copy()
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

X_train = X_train.reshape(-1,28,28,1)
X_test  = X_test.reshape(-1,28,28,1)


# model
model_ = Sequential()
model_.add(Conv2D(32, (24,24), input_shape=(28, 28, 1), activation='relu'))
model_.add(MaxPooling2D(pool_size=(2, 2)))
model_.add(Dropout(0.4))
model_.add(Flatten())
model_.add(Dense(128, activation='relu'))
model_.add(Dense(18, activation='softmax'))


model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=200, verbose=2)

scores = model_.evaluate(X_test,Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

from sklearn.metrics import accuracy_score
y_pred = model_.predict_classes(X_test)
acc = accuracy_score(Y_test_for_accuracy_matrix, y_pred)
print(f"Accuracy: {acc}")



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
