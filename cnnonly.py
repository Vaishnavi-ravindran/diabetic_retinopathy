# =========================================
# STEP 1: Imports
# =========================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# =========================================
# STEP 2: Paths
# =========================================
train_dir = r"C:\Users\HP\Documents\github\diabetic_retinopathy\new_dataset\split_dataset\train"
test_dir  = r"C:\Users\HP\Documents\github\diabetic_retinopathy\new_dataset\split_dataset\test"

output_dir = r"C:\Users\HP\Documents\github\diabetic_retinopathy\output"
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# =========================================
# STEP 3: Data Generators
# =========================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Classes:", train_generator.class_indices)

# =========================================
# STEP 4: Class Weights (IMPORTANT)
# =========================================
from sklearn.utils import class_weight

y_train = train_generator.classes

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =========================================
# STEP 5: MODEL (ResNet50)
# =========================================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.6)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# =========================================
# STEP 6: FOCAL LOSS
# =========================================
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        return alpha * (1 - pt) ** gamma * ce
    return loss

# =========================================
# STEP 7: COMPILE
# =========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

model.summary()

# =========================================
# STEP 8: TRAINING
# =========================================
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
]

print("\nTraining Model...")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save model
model.save(os.path.join(output_dir, "cnn_resnet50.h5"))

# =========================================
# STEP 9: EVALUATION
# =========================================
print("\nFinal Results:")
loss, acc = model.evaluate(test_generator)
print("Accuracy:", acc)

# Predictions
y_pred = np.argmax(model.predict(test_generator), axis=1)
y_true = test_generator.classes

# =========================================
# STEP 10: REPORT
# =========================================
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# =========================================
# STEP 11: CONFUSION MATRIX
# =========================================
def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(path)
    plt.show()

plot_confusion(y_true, y_pred, os.path.join(output_dir, "cnn_cm.png"))

# =========================================
# STEP 12: KAPPA SCORE
# =========================================
print("\n📊 Cohen Kappa Score:")
print(cohen_kappa_score(y_true, y_pred))

# =========================================
# STEP 13: GRAD-CAM (FIXED)
# =========================================

import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):

    # Get ResNet base model inside your model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# =========================================
# GET ONE IMAGE FROM TEST GENERATOR
# =========================================
img, _ = next(test_generator)   # get batch
img = img[0]                   # first image
img_array = np.expand_dims(img, axis=0)

# Generate heatmap
heatmap = make_gradcam_heatmap(img_array, model)

# Plot
plt.imshow(img)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Overlay")
plt.axis('off')
plt.savefig(os.path.join(output_dir, "gradcam.png"))
plt.show()