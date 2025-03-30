import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# ---- GOOGLE DRIVE INTEGRATION ----


# ---- DEFINE DIRECTORIES ----
#dataset_dir = '/kaggle/input/plzzzzzzzz/SKINDATASETLIMITEDbutgay'  # Change to your dataset path
checkpoint_path = "/kaggle/working/checknigger.keras"
joblib_path = "skin_disease_model.keras"
img_size = (128, 128)  # CHANGED FROM (720, 1280)
# Load metadata CSV
#df = pd.read_csv("/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv")

# Add proper file extensions (images are .jpg but filenames in CSV lack extension)
#df['image_id'] = df['image_id'] + '.jpg'



# Point to your image directory (should contain all JPG files)
dataset_dir = "/kaggle/working/patotes/a7a"

datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=10, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1
)

# ---- LOAD TRAINING & VALIDATION DATA ----
train_generator = datagen.flow_from_directory(
    dataset_dir, target_size=img_size,  # Now using 224x224
    batch_size=16, class_mode="categorical", subset="training",
    
    color_mode='grayscale'
)

val_generator = datagen.flow_from_directory(
    dataset_dir, target_size=img_size,  # Now using 224x224
    batch_size=16, class_mode="categorical", subset="validation",
    color_mode='grayscale'
)
 #---- CNN MODEL ARCHITECTURE ----
model = Sequential()

# Convolutional Block 1 (Input shape updated)
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(128, 128, 1)))
  # CHANGED
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

  # CHANGED
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))


# Convolutional Block 2
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# Fully Connected Layers
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

# Compile the model (unchanged)
model.compile(optimizer=Adam(learning_rate=0.00001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Print model summary
model.summary()

# ---- CUSTOM CALLBACK FOR EPOCH PROGRESS ----
class EpochProgressIndicator(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch + 1}/{self.params['epochs']}")


# ---- CALLBACKS ----
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,  # Now only saves the best model
    mode='min',
    verbose=1
)

# ---- CHECK IF A SAVED MODEL EXISTS ----
if os.path.exists(checkpoint_path):
    print("Loading saved model...")
    model = load_model(checkpoint_path)  # Load the single checkpoint file
else:
    print("No saved model found. Training from scratch.")

# Continue training (no changes to the rest)
early_stopping = EarlyStopping(
    monitor='val_loss',       # Monitor validation loss
    patience=5,               # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1                 # Print a message when stopping early
)
history = model.fit(
    train_generator,  # Fixed variable name
    validation_data=val_generator,  # Fixed variable name
    epochs=100,
    callbacks=[early_stopping, checkpoint],  # Fixed callback reference
    # Removed undefined class_weight parameter
)

model.save("final_modeman.keras")  # Consistent save name
