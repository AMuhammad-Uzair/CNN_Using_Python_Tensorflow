#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 02:41:11 2021

@author: muhammaduzair
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir="Data/Training_Data"
data_dir1="Data/Testing_Data"

#Reading Data from folder
Train_Data=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(100,
    100), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False
)

#Reading Data from folder
Test_Data=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir1, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(100,
    100), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False
)


Train_labels = Train_Data.class_names
Test_labels = Test_Data.class_names


trainX=[]
trainY=[]
plt.figure(figsize=(10, 10))
for images, labels in Train_Data.take(500):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    trainX.append(images[i].numpy().astype("uint8"))
    trainY.append(Train_labels[labels[i]])
    
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(Train_labels[labels[i]])
    plt.axis("off")


testX=[]
testY=[]
for images, labels in Test_Data.take(500):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    testX.append(images[i].numpy().astype("uint8"))
    testY.append(Test_labels[labels[i]])
    
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(Test_labels[labels[i]])
    plt.axis("off")

trainX=np.array(trainX)
trainX=trainX.reshape(3951,100,100,3)

testX=np.array(testX)
testX=testX.reshape(846,100,100,3)

# Normalize pixel values to be between 0 and 1
train_images, test_images = trainX / 255.0, testX / 255.0
test_labels=testY
train_labels=trainY


Train_Labels=[]

for i in trainY:
    if i=="buildings":
        Train_Labels.append(1)
    elif i=="forest":
        Train_Labels.append(2)    
    elif i=="glacier":
        Train_Labels.append(3)   
    elif i=="mountain":
        Train_Labels.append(4)   
    elif i=="sea":
        Train_Labels.append(5)   
    elif i=="street":
        Train_Labels.append(6)   
        
Train_Labels=np.array(Train_Labels)

Test_Labels=[]

for j in testY:
    if j=="buildings":
        Test_Labels.append(1)
    elif j=="forest":
        Test_Labels.append(2)    
    elif j=="glacier":
        Test_Labels.append(3)   
    elif j=="mountain":
        Test_Labels.append(4)   
    elif j=="sea":
        Test_Labels.append(5)   
    elif j=="street":
        Test_Labels.append(6)  


Test_Labels=np.array(Test_Labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, Train_Labels, epochs=10, 
                    validation_data=(test_images, Test_Labels))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  Test_Labels, verbose=2)


print(test_acc)









