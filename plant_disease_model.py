import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0, Xception, DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  Enable Mixed Precision for GPU Efficiency
set_global_policy('mixed_float16')

print("🔹 Loading Dataset...")
data_dir = "C:\\Users\\ashwi\\GUVI_Projects\\PlantDisease\\Dataset"
train_dir, val_dir = os.path.join(data_dir, "train"), os.path.join(data_dir, "valid")

#  Define Training Parameters
image_size = (96, 96) 
BATCH_SIZE = 16  
AUTOTUNE = tf.data.AUTOTUNE 

#  Load Datasets with Prefetching for Faster Training
train_dataset = image_dataset_from_directory(
    train_dir, image_size=image_size, batch_size=BATCH_SIZE, label_mode='int'
).shuffle(1000).cache().prefetch(AUTOTUNE)

val_dataset = image_dataset_from_directory(
    val_dir, image_size=image_size, batch_size=BATCH_SIZE, label_mode='int'
).cache().prefetch(AUTOTUNE)

#  Extract Class Names
num_classes = len(os.listdir(train_dir))

#  Define Pretrained Models for Transfer Learning
models_to_compare = {
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(96, 96, 3)),
    "Xception": Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
}

model_results = {}

for model_name, base_model in models_to_compare.items():
    print(f"\n **Training {model_name} Model...**")
    base_model.trainable = False  # Freeze base model layers

    #  Define Custom Model Architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    #  Compile Model
    model.compile(optimizer=Adam(learning_rate=0.0001),  
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #  Training with Callbacks
    callbacks = [
        ModelCheckpoint(f'{model_name}_best.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    #  Train Model
    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)

    #  Store Trained Model
    model_results[model_name] = model

    #  Free Memory
    K.clear_session()

#  Step 7: Evaluate Models on Validation Dataset
performance_metrics = {}

# Extract ground truth labels from validation dataset
y_true = np.concatenate([y.numpy() for _, y in val_dataset])

for model_name, model in model_results.items():
    print(f"\n Evaluating {model_name} on Validation Dataset...")

    # Get predictions
    val_predictions = model.predict(val_dataset)
    y_pred = np.argmax(val_predictions, axis=1)

    # Compute Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    performance_metrics[model_name] = (accuracy, precision, recall, f1)

#  Step 8: Compare Model Performance
df_performance = pd.DataFrame(performance_metrics, index=["Accuracy", "Precision", "Recall", "F1-score"]).T
print("\n🏆 **Best Performing Model:**", df_performance["Accuracy"].idxmax())

#  Save the Best Model
best_model_name = df_performance["Accuracy"].idxmax()
best_model = model_results[best_model_name]
best_model.save("best_plant_disease_model.keras")
print(f" Best model saved as `best_plant_disease_model.keras`")

#  Display Performance Summary
print("\n **Model Performance Summary:**")
print(df_performance)

#  Clear Memory
K.clear_session()
