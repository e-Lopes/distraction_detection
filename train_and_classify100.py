import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils import class_weight

# === GPU CONFIGURATION CHECK ===
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)  # Log GPU/CPU ops (optional)

# Ensure TensorFlow is using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to use only the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU detected. Using:", gpus[0])
    except RuntimeError as e:
        print("GPU setup error:", e)
else:
    print("No GPU found. Falling back to CPU.")

# === CONFIGURATION ===
base_dir = '/home/edu/Desktop/distraction_detection/distraction_detection/frames'
ok_dir = os.path.join(base_dir, 'OK')
nok_dir = os.path.join(base_dir, 'NOK')
ok_ia_dir = os.path.join(base_dir, 'OK_IA')
nok_ia_dir = os.path.join(base_dir, 'NOK_IA')

os.makedirs(ok_ia_dir, exist_ok=True)
os.makedirs(nok_ia_dir, exist_ok=True)

img_size = (224, 224)
batch_size = 32
epochs = 10

# === DATA AUGMENTATION (GPU-OPTIMIZED) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['NOK', 'OK'],
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['NOK', 'OK'],
    subset='validation',
    shuffle=False
)

# === GPU-OPTIMIZED MODEL ===
with tf.device('/GPU:0'):  # Explicitly place the model on GPU
    base_model = EfficientNetB0(
        include_top=False,
        input_shape=(*img_size, 3),
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )

# === TRAINING (GPU-ACCELERATED) ===
print("\nStarting training on GPU...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1
)

# === CLASSIFICATION (GPU-BATCHED) ===
def classify_with_gpu(image_paths, batch_size=32):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = load_img(path, target_size=img_size)
            img = img_to_array(img) / 255.0
            batch_images.append(img)
        
        batch_images = np.array(batch_images)
        probs = model.predict(batch_images, verbose=0)
        
        for path, prob in zip(batch_paths, probs):
            prob = prob[0]
            destino = ok_ia_dir if prob >= 0.5 else nok_ia_dir
            shutil.move(path, os.path.join(destino, os.path.basename(path)))
            print(f"[{'OK' if prob >= 0.5 else 'NOK'}] {os.path.basename(path)} → {prob:.4f}")

# Process images in GPU-friendly batches
restantes = [os.path.join(base_dir, f) for f in os.listdir(base_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
classify_with_gpu(restantes[:100])