import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries

# --- 1. Data Loading and Parameters ---
# This section is based on your provided code snippet.
# NOTE: The original paper used 4 classes, while this setup uses 3.
# The image size and training parameters have also been adjusted as per your request.

DATA_DIR = '/Users/ivan/Documents/Repos/xai-gls/chestRays/data/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model and training parameters
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.00001
CLASS_NAMES = ['NORMAL', 'COVID19', 'PNEUMONIA']
NUM_CLASSES = len(CLASS_NAMES)

# Create TensorFlow Datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical', # Use 'categorical' for one-hot encoded labels
    class_names=CLASS_NAMES,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.15,
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

# Normalize images to [0, 1] range
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(normalize)
validation_dataset = validation_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


# --- 2. Proposed CNN Model Architecture (Adapted) ---
# The architecture from the paper is adapted for the new input shape and number of classes.
def create_cnn_model(input_shape, num_classes):
    """
    Creates the lightweight CNN model described in the paper.
    """
    inputs = Input(shape=input_shape)

    # [cite_start]Convolutional Blocks from the paper [cite: 159, 160]
    # Block 1
    x = Conv2D(filters=3, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.05)(x)

    # Block 2
    x = Conv2D(filters=96, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.2)(x)

    # Block 3
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    
    # Block 4
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    # Flattening and Dense Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.45)(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x) # Adjusted for num_classes

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- 3. Model Training ---
# [cite_start]NOTE: The paper uses 10-fold cross-validation[cite: 35].
# This script performs a single training run based on the train/validation split provided.
if(not os.path.exists('my_model.keras')):
    model = create_cnn_model(IMG_SHAPE, NUM_CLASSES)

    # Compile the model with the specified learning rate and categorical loss
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy', # Changed for one-hot labels
                metrics=['accuracy'])

    model.summary()

    # [cite_start]Early stopping callback as mentioned in the paper [cite: 280]
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[early_stopping]
    )
    model.save('my_model.keras')
else:
    model = tf.keras.models.load_model('my_model.keras')

# --- 4. Model Evaluation ---
print("\n--- Evaluating Model on Test Data ---")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

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
preds = model.predict(sample_image)
print(f"Predicted Label: {CLASS_NAMES[np.argmax(preds)]} with probability {np.max(preds):.4f}")


# 5.1 SHAP (SHapley Additive exPlanations)
explainer_shap = shap.GradientExplainer(model, background_images.numpy())
shap_values = explainer_shap.shap_values(sample_image.numpy())

print("\nDisplaying SHAP explanation plot...")
shap.image_plot(shap_values, -sample_image.numpy(), show=False)
plt.suptitle("SHAP Explanation")
plt.savefig('shap01.png')

# 5.2 LIME (Local Interpretable Model-agnostic Explanations)
explainer_lime = lime.lime_image.LimeImageExplainer()
explanation_lime = explainer_lime.explain_instance(
    sample_image[0].numpy().astype('double'),
    model.predict,
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
last_conv_layer_name = [layer.name for layer in model.layers if "conv2d" in layer.name][-1]
heatmap = make_gradcam_heatmap(sample_image, model, last_conv_layer_name)

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