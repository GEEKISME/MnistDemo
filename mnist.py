from keras.utils import to_categorical
from keras import models,layers,regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
# print(train_images.shape,test_images.shape)
# print("======================================")
# print(train_images[0])
# print("======================================")
# print(train_labels[0])
# print("======================================")
# plt.imshow(train_images[0])
# plt.show()
train_images = train_images.reshape((60000,28*28)).astype('float')
test_images = test_images.reshape((10000,28*28)).astype('float')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels[0])

# ========================================================================method 1
# #  建立模型
# network = models.Sequential()
# # Dense就是密集的，全连接的意思
# network.add(layers.Dense(units=15,activation='relu',input_shape=(28*28, ),))
# network.add(layers.Dense(units=10,activation='softmax'))
#
# # 编译步骤
# network.compile(optimizer=RMSprop(lr = 0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# # 训练网络 用fit函数，epochs表示训练多少个会和，batch——size表示每次训练给多大的数据
# network.fit(train_images,train_labels,epochs=20,batch_size=128,verbose=2)
#
# print(network.summary())
#
# y_pre = network.predict(test_images[:5])
# print(y_pre,test_labels[:5])
# testloss, test_accuracy = network.evaluate(test_images,test_labels)
# print("testloss:",testloss,"      test_accuracy:",test_accuracy)

# ============================================================= method 2
#  建立模型
network = models.Sequential()
# Dense就是密集的，全连接的意思
network.add(layers.Dense(units=128,activation='relu',input_shape=(28*28, ),
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=32,activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=10,activation='softmax'))
# 编译步骤
network.compile(optimizer=RMSprop(lr = 0.001),loss='categorical_crossentropy',metrics=['accuracy'])
# 训练网络 用fit函数，epochs表示训练多少个会和，batch——size表示每次训练给多大的数据
network.fit(train_images,train_labels,epochs=20,batch_size=128,verbose=2)

print(network.summary())

y_pre = network.predict(test_images[:5])
print(y_pre,test_labels[:5])
testloss, test_accuracy = network.evaluate(test_images,test_labels)
print("testloss:",testloss,"      test_accuracy:",test_accuracy)