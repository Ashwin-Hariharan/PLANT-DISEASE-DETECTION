import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

# Dataset path
dataset_path = r"C:\Users\ashwi\GUVI_Projects\PlantDisease\Dataset\balanced"
save_model_path = r"C:\Users\ashwi\GUVI_Projects\PlantDisease\SavedModels"
os.makedirs(save_model_path, exist_ok=True)

# Model-specific preprocessing functions
preprocess_functions = {
    'ResNet50': resnet_preprocess,
    'DenseNet121': densenet_preprocess,
    'EfficientNetB0': efficientnet_preprocess
}

# Define a function to use a pre-trained model with a custom classifier
def use_pretrained_model(model_class, preprocess_function, model_name):
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.1
    )

    train_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = data_gen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    
    base_model = model_class(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Keep feature extractor frozen
    
    global_avg_pool = GlobalAveragePooling2D()(base_model.output)
    dense = Dense(256, activation='relu')(global_avg_pool)
    dropout = Dropout(0.5)(dense)
    output = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[early_stopping]
    )
    
    model.save(os.path.join(save_model_path, f'{model_name}.h5'))
    
    return history

# Use pre-trained models and store histories
models = {
    'ResNet50': ResNet50,
    'DenseNet121': DenseNet121,
    'EfficientNetB0': EfficientNetB0
}

histories = {}
for model_name, model_class in models.items():
    print(f'Using pre-trained {model_name} with custom classifier and preprocessing...')
    histories[model_name] = use_pretrained_model(model_class, preprocess_functions[model_name], model_name)

# Plot accuracy and loss
plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{model_name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{model_name} Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
