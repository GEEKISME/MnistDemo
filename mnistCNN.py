from keras.utils import to_categorical
from keras import models,layers,regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

# 搭建LeNet网络
def LeNet():
    network = models.Sequential
    network.add(layers.Conv2D(filters=6,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
    network.add(layers.AveragePooling2D((2,2)))
    network.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'))
    network.add(layers.AveragePooling2D((2,2)))
    network.add(layers.Conv2D(filters=120,kernel_size=(3,3),activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(84,activation='relu'))
    network.add(layers.Dense(10,activation='softmax'))
    return network

network = LeNet()
network.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
train_images = train_images.reshape((60000,28*28)).astype('float')
test_images = test_images.reshape((10000,28*28)).astype('float')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images,train_labels,epochs=10,batch_size=128,verbose=2)
test_loss,test_accuracy = network.evaluate(test_images,test_labels)