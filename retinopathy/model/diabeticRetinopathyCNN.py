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

import shap
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

if(not os.path.exists('my_model.keras')):
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
    proposed_model.save('my_model.keras')
    # --- 6. EVALUATION ---
    print("\nEvaluating the Proposed Model...")

    # Plotting accuracy and loss curves
    acc = proposed_model.history['accuracy']
    val_acc = proposed_model.history['val_accuracy']
    loss = proposed_model.history['loss']
    val_loss = proposed_model.history['val_loss']

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
    plt.savefig('training_validation_loss.png')

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
else:
    proposed_model = tf.keras.models.load_model('my_model.keras')
    history = proposed_model.history



# --- 5. Model Explanation with XAI ---
# We need to extract a sample image and a background dataset for the explainers.

# Extract a batch from the test dataset for explanation
test_images, test_labels = next(iter(test_dataset))
sample_image = test_images[0:1] # Get the first image in the batch
sample_label = test_labels[0]

# Extract a batch from the training dataset for SHAP background
background_images, _ = next(iter(train_dataset.take(1)))

print(f"\n--- Generating XAI Explanations for a sample image ---")
print(f"True Label: {CLASS_NAMES[np.argmax(sample_label)]}")
preds = proposed_model.predict(sample_image)
print(f"Predicted Label: {CLASS_NAMES[np.argmax(preds)]} with probability {np.max(preds):.4f}")


# 5.1 SHAP (SHapley Additive exPlanations)
explainer_shap = shap.GradientExplainer(proposed_model, background_images.numpy())
shap_values = explainer_shap.shap_values(sample_image.numpy())

print("\nDisplaying SHAP explanation plot...")
shap.image_plot(shap_values, -sample_image.numpy(), show=False)
plt.suptitle("SHAP Explanation")
plt.savefig('shap01.png')

# 5.2 LIME (Local Interpretable Model-agnostic Explanations)
explainer_lime = lime.lime_image.LimeImageExplainer()
explanation_lime = explainer_lime.explain_instance(
    sample_image[0].numpy().astype('double'),
    proposed_model.predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

print("\nDisplaying LIME explanation plot...")
temp, mask = explanation_lime.get_image_and_mask(
    explanation_lime.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis('off')
plt.savefig('lime01.png')

# 5.3 Grad-CAM (Gradient-weighted Class Activation Mapping)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Find the name of the last convolutional layer
last_conv_layer_name = [layer.name for layer in proposed_model.layers if "conv2d" in layer.name][-1]
heatmap = make_gradcam_heatmap(sample_image, proposed_model, last_conv_layer_name)

print("\nDisplaying Grad-CAM heatmap...")
plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.savefig('gradcam01.png')

# Superimpose the heatmap on the original image
img = sample_image[0].numpy()
heatmap = np.uint8(255 * heatmap)
jet = plt.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

plt.imshow(superimposed_img)
plt.title("Grad-CAM Superimposed")
plt.axis('off')
plt.savefig('gradcam_superimposed01.png')