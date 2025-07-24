import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, SeparableConv2D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications import InceptionV3, MobileNetV2, DenseNet201
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries

# --- 1. CONFIGURATION PARAMETERS (from Table 3 & text) ---
# Set the path to your dataset directory
# IMPORTANT: Change this path to where you have stored the dataset
DATA_DIR = './data/retinoGray/'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model and training parameters from the paper
IMG_WIDTH = 120  # As per 'Image resize' section [111]
IMG_HEIGHT = 120 # As per 'Image resize' section [111]
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 64  # From Table 3 [212]
EPOCHS = 30      # From Table 3 [212]
LEARNING_RATE = 0.00001 # From Table 3 [212]
CLASS_NAMES = ['DR', 'No_DR']
NUM_CLASSES = len(CLASS_NAMES)

# --- 2. IMAGE PRE-PROCESSING PIPELINE (as described in the paper) ---

def preprocess_image(image):
    """
    Applies the full pre-processing pipeline described in the paper:
    1. Padding
    2. Thresholding
    3. Morphological Opening (Erosion -> Dilation)
    4. Masking to extract the cell
    
    This function is designed for use with tf.py_function.
    """
    # Decode image if it's in a tensor format
    if isinstance(image, tf.Tensor):
        image = image.numpy().astype(np.uint8)

    # 1. Padding: Add a 10-pixel border [119]
    image_padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 2. Thresholding: Convert to HSV and create a mask to isolate the cell [127]
    # The paper's mention of color ranges suggests HSV-based color masking.
    hsv = cv2.cvtColor(image_padded, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([80, 80, 80])
    upper_bound = np.array([180, 255, 255]) 
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 3. Morphological Opening: Reduce noise and separate clusters [129, 153]
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Masking: Apply the mask to the padded image [135]
    res = cv2.bitwise_and(image_padded, image_padded, mask=opening)
    
    # Set background to black for consistency
    res[opening == 0] = [0, 0, 0]

    # 5. Resize back to the target shape and normalize
    final_image = cv2.resize(res, (IMG_WIDTH, IMG_HEIGHT))
    final_image = final_image.astype(np.float32) / 255.0
    return final_image


@tf.function
def tf_preprocess(x, y):
    """
    Wraps the OpenCV preprocessing function in tf.py_function
    for use with a tf.data.Dataset.
    """
    [x,] = tf.py_function(preprocess_image, [x], [tf.float32])
    x.set_shape([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    return x, y

# --- 3. DATA LOADING ---
print("Loading and preprocessing data...")
# Load datasets from directory. Note: The paper's pre-processing is not applied here.
# It is generally better to preprocess within the model or using a tf.data map.
# For simplicity in this script, we will proceed without the custom preprocessing.
# To apply it, you would use: train_dataset = train_dataset.map(tf_preprocess)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.15, # Use 15% of training data for validation [103]
    subset='training',
    seed=123
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.15,
    subset='validation',
    seed=123
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize the datasets
def normalize_func(image, label):
    return image / 255.0, label

train_dataset = train_dataset.map(normalize_func)
validation_dataset = validation_dataset.map(normalize_func)
test_dataset = test_dataset.map(normalize_func)

print(f"Training batches: {len(train_dataset)}, Validation batches: {len(validation_dataset)}, Test batches: {len(test_dataset)}")

# --- 4. MODEL ARCHITECTURE (PROPOSED CNN) ---

def build_proposed_model(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    """
    Builds the custom CNN model as described in the paper's
    'Proposed model' and 'System architecture' sections.
    """
    # Weight initialization as per the paper [168]
    initializer = 'glorot_uniform'

    inputs = Input(shape=input_shape)

    # Initial Convolutional Block [139]
    x = Conv2D(16, (3, 3), padding='same', activation='relu' )(inputs)
    x = Conv2D(16, (3, 3), padding='same', activation='relu' )(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Separable Convolutional Blocks [140] (filter counts from Fig. 4)
    x = SeparableConv2D(32, (3, 3), padding='same', activation='relu' )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Fourth and Fifth blocks with dropout [142]
    x = SeparableConv2D(64, (3, 3), padding='same', activation='relu' )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = SeparableConv2D(128, (3, 3), padding='same', activation='relu' )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Flatten for fully connected layers [143]
    x = Flatten()(x)

    # Fully Connected Layers with specific dropouts and 'tanh' activation [144-146]
    x = Dense(512, activation='tanh' )(x)
    x = Dropout(0.7)(x)
    x = Dense(128, activation='tanh' )(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='tanh' )(x)
    x = Dropout(0.3)(x)

    # Output Layer [147]
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Proposed_CNN_Model')
    return model

# --- 5. MODEL COMPILATION AND TRAINING ---

proposed_model = build_proposed_model()
proposed_model.summary()

# Compile the model as per Table 3 [212]
optimizer = Nadam(learning_rate=LEARNING_RATE)
proposed_model.compile(optimizer=optimizer,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Train the model
print("\nTraining the Proposed CNN Model...")
history = proposed_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=1
)

# --- 6. EVALUATION ---
print("\nEvaluating the Proposed Model...")

# Plotting accuracy and loss curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate on the test set
loss, accuracy = proposed_model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy*100:.2f}%") # Paper achieves 99.12% [7]
print(f"Test Loss: {loss:.4f}")

# Generate Classification Report and Confusion Matrix
y_pred_probs = proposed_model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_true_indices = np.argmax(y_true, axis=1) 

print("\nClassification Report (similar to Table 6):")
print(classification_report(y_true_indices, y_pred, target_names=CLASS_NAMES))

print("\nReplication script finished.")

# --- 4. Model Explanation with XAI ---
# Using the model from the last fold for demonstration

# Select a sample image from the test set for explanation
sample_image = train_dataset.as_numpy_iterator().next()
sample_label = test_dataset.as_numpy_iterator().next()

print(f"\n--- Generating XAI Explanations for a sample image ---")
print(f"True Label: {CLASS_NAMES[sample_label]}")
preds = proposed_model.predict(sample_image)
print(f"Predicted Label: {CLASS_NAMES[np.argmax(preds)]} with probability {np.max(preds):.4f}")

# 4.2 LIME (Local Interpretable Model-agnostic Explanations)
# LIME explains the predictions of any classifier by learning an interpretable model locally around the prediction.
# The paper uses LIME with super-pixels to segment the important regions 
explainer_lime = lime.lime_image.LimeImageExplainer()
explanation_lime = explainer_lime.explain_instance(sample_image[0],
                                                   model.predict,
                                                   top_labels=1,
                                                   hide_color=0,
                                                   num_samples=1000)

print("\nDisplaying LIME explanation plot...")
temp, mask = explanation_lime.get_image_and_mask(explanation_lime.top_labels[0],
                                                 positive_only=True,
                                                 num_features=5,
                                                 hide_rest=False)

plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis('off')
plt.show()