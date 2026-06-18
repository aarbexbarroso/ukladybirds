import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


RUN = 5
SEED = 41+RUN
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


TRAIN_DIR = './Images/Batch_TVT'+str(RUN)+'/train'
VAL_DIR   = './Images/Batch_TVT'+str(RUN)+'/validation'
TEST_DIR   = './Images/Batch_TVT'+str(RUN)+'/test'
CSV_OUTPUT_DIR = "./csv_10/Resnet152_256/Batch"+str(RUN)


os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_EPOCHS = 500
LEARNING_RATE = 0.0001


def preprocess_resnet(img):
    return preprocess_input(img)


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_resnet,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.5, 1.5),
    horizontal_flip=True,
    fill_mode="wrap"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_resnet
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)



num_classes = train_generator.num_classes


base_model = ResNet152(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
base_model.trainable = False  


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


CALLBACKS = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1, min_delta=0.01),
    ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)
]


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=NUM_EPOCHS,
    callbacks=CALLBACKS
)


model.save('resnet152_256_final_batch_'+str(RUN)+'.h5')
print("\nModel saved as 'resnet152_256_final_batch_'+str(RUN)+'.h5'")


def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_training(history)


def generate_predictions_csv(generator, model, output_csv, dataset_name):
    predictions = model.predict(generator, verbose=1)
    y_true = generator.classes
    class_labels = list(generator.class_indices.keys())
    image_ids = [os.path.basename(filepath) for filepath in generator.filepaths]
    df = pd.DataFrame(predictions, columns=class_labels)
    df.insert(0, 'real_class', [class_labels[i] for i in y_true])
    df.insert(0, 'image_id', image_ids)
    csv_path = os.path.join(CSV_OUTPUT_DIR, output_csv)
    df.to_csv(csv_path, index=False)
    print(f"{dataset_name} CSV saved at {csv_path}")


generate_predictions_csv(val_generator, model, "val_predictions_batch_"+str(RUN)+".csv", "Validation")
generate_predictions_csv(test_generator, model, "test_predictions_batch_"+str(RUN)+".csv", "Test")









