# train_fruit_classifier.py
# Train a multi-class CNN for fruit disease classification

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json

# 1. Data Setup
DATA_DIR = "/Users/momo/Downloads/c++/Desktop Application Development/the proper app/ML Project/extracted_data"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# 2. Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class labels
os.makedirs('saved_model', exist_ok=True)
with open('saved_model/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# 3. Model Creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# 5. Evaluation
Y_pred = model.predict(val_generator, val_generator.samples // BATCH_SIZE + 1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes[:len(y_pred)]
labels = list(train_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 6. Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("saved_model/training_plots.png")
plt.show()

# 7. Save the model
model.save("saved_model/fruit_classifier_model")  # folder-based, GitHub-friendly
print("\n Model and artifacts saved to 'saved_model/'")
import json
with open("saved_model/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("\n Class indices saved to 'saved_model/class_indices.json'")
