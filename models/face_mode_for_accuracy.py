import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Directories for training and testing data
train_dir = "/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/input/fer2013/train" # Directory containing the training data in linux
test_dir = "/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/input/fer2013/test"  # Directory containing the validation data in linux

train_datagen = ImageDataGenerator(
    width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
    height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
    horizontal_flip = True,         # Flip images horizontally at random
    rescale = 1./255,               # Rescale pixel values to be between 0 and 1
    validation_split = 0.2          # Set aside 20% of the data for validation
    
)


validation_datagen = ImageDataGenerator(
    rescale = 1./255,               # Rescale pixel values to be between 0 and 1
    validation_split = 0.2          # Set aside 20% of the data for validation
)


# ImageDataGenerator for data augmentation
train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "rgb",  # Changed from "grayscale" to "rgb"
    class_mode = "categorical",
    subset = "training"
)

validation_generator = validation_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "rgb",  # Changed from "grayscale" to "rgb"
    class_mode = "categorical",
    subset = "validation"
)

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

model = Sequential()
model.add(vgg_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Define the checkpoint
checkpoint_callback = ModelCheckpoint(
    filepath='model_weights_vgg16.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint_callback]
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], 'bo', label='Training accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], 'bo', label='Training loss')
plt.plot(history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()

# Confusion matrix
sns.set_theme()
validation_labels = validation_generator.classes
validation_pred_probs = model.predict(validation_generator)
validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
class_names = list(train_generator.class_indices.keys())
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Display a few samples of predictions
sample_images, sample_labels = next(validation_generator)
sample_preds = model.predict(sample_images)
sample_preds_labels = np.argmax(sample_preds, axis=1)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_images[i].reshape(48, 48), cmap='gray')
    plt.title(f"True: {class_names[sample_labels[i].argmax()]}\nPred: {class_names[sample_preds_labels[i]]}")
    plt.axis('off')
plt.savefig('sample_predictions.png')
plt.show()
