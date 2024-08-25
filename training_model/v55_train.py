from __future__ import print_function
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

num_classes = 7 #five emotions taken so five classes
img_rows,img_cols = 48,48 
batch_size = 32 #we have choosen 32 as batch size, as we are using 32 image at a time, to save computation time

train_data_dir = r'C:\\laragon\\www\\8th_sem_project\\emotion-detection-and-music-recommendation\\dataset\\input\\fer2013\\train'
validation_data_dir = r'C:\\laragon\\www\\8th_sem_project\\emotion-detection-and-music-recommendation\\dataset\\input\\fer2013\\test'

#ImageDataGenerator-> we can generate some image if we have less amount of image 
#by performing operations like rotate left, right, zoom out, zoom in, left shift, right shift 
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

#since we have multiple classes hence Class_mode=categorical
train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


model = Sequential()

# Block-1

#he_normal- weights, by trun normal distribution centered at 0
#"same" results in padding the input such that the output has the same length as the original input 
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#model.add(Dropout(0.2))

# Block-4 one more layer of 256 added

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#model.add(Dropout(0.2))

# Block-5

#we dont want matrix so it will be flatten and we will get a 1-D array
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))


print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#Checkpoints- For saving the best model with min val_loss
checkpoint = ModelCheckpoint('Emotion_little_vgg.keras',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

#earlystop- if there is no improvement in the val_accuracy the model will stop
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=90,
                          verbose=1,
                          restore_best_weights=True
                          )
#reduce_lr- it will reduce the learning rate with factor 0.2
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=30,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(learning_rate=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=100

history = model.fit(
    train_generator,
    steps_per_epoch=28709 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // batch_size,
    callbacks=callbacks  # Ensure this line is properly indented and formatted
)

# 1. Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_performance.png')
plt.show()

# 2. Generate Confusion Matrix
Y_pred = model.predict(validation_generator, nb_validation_samples // batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true classes
y_true = validation_generator.classes

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# 3. Generate Classification Report
report = classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys())
print(report)

# Save the classification report to a file
with open('classification_report.txt', 'w') as f:
    f.write(report)
    
model.save('Emotion_little_vgg.keras')
