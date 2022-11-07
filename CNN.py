from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
from matplotlib import pyplot
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

#reshaping data
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], train_X.shape[2], 1))
test_X = test_X.reshape((test_X.shape[0],test_X.shape[1],test_X.shape[2],1)) 
#checking the shape after reshaping
print(train_X.shape)
print(test_X.shape)

#normalizing the pixel values
train_X=train_X/255
test_X=test_X/255

#defining model
model=Sequential()
#adding convolution layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#adding output layer
model.add(Dense(10,activation='softmax'))
#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model.fit(train_X,train_y,epochs=10)

#evaluting the model
model.evaluate(test_X,test_y)

#predict first 4 images in the test set
print(model.predict(test_X[:4]))


#actual results for first 4 images in test set
print(test_y[:4])



 





