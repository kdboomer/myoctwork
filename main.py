# Import Libraries

import os
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D



train_path = r'C:\Users\admin\Desktop\project\dateset\train'
test_path = r'C:\Users\admin\Desktop\project\dateset\test'


os.listdir(train_path)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


os.listdir(test_path)


# Path to the directory containing the class folders
train_data = train_path

# List of classes
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Set up a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Display one image from each class in the grid
for i, class_name in enumerate(classes):
    # Get the list of files in the class folder
    class_folder = os.path.join(train_data, class_name)
    files = os.listdir(class_folder)

    # Display the first image in the class folder in the corresponding grid cell
    if files:
        img_path = os.path.join(class_folder, files[0])
        img = mpimg.imread(img_path)

        axs[i // 2, i % 2].imshow(img)
        axs[i // 2, i % 2].set_title(class_name)
        axs[i // 2, i % 2].axis('off')

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.show()


# Path to the directory containing the class folders
train_data = train_path

# List of classes
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Create an empty dictionary to store the count of images in each class
class_counts = {}

# Count the number of images in each class
for class_name in classes:
    class_folder = os.path.join(train_data, class_name)
    files = os.listdir(class_folder)
    class_counts[class_name] = len(files)

# Convert the dictionary to a pandas DataFrame for better display
class_counts_df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])

# Display the DataFrame
print("Class Value Counts:")
print(class_counts_df)

# Plot a bar chart for better visualization
plt.bar(class_counts_df['Class'], class_counts_df['Count'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()



# Image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 512

# Using ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split 20% of the data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  # 4 classes


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


num_epochs = 10

history = model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)

model.save('my_model.h5')


# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')