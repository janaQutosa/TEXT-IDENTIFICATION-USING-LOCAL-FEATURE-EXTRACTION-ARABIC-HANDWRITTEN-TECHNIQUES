# Shiyar Mohammad - 1210766
# Jana Qutosa - 1210331
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# _______________________Function to load dataset_______________________

def load_dataset(dataset_path, img_size=(128, 128)):
    data = []
    labels = []
    for user_id in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user_id)
        if os.path.isdir(user_path):
            for img_file in os.listdir(user_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(user_path, img_file)
                    img = Image.open(img_path).convert("L")
                    img = img.resize(img_size)
                    data.append(np.array(img))
                    labels.append(user_id)
    return np.array(data), np.array(labels)

# _______________________Custom CNN Model_______________________

def build_custom_cnn(input_shape, num_classes, filters=[32, 64], dense_units=128, dropout_rate=0.5):
    model = Sequential()

    # Add convolutional blocks based on filter sizes
    for filter_size in filters:
        model.add(Conv2D(filter_size, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        input_shape = None  # Input shape only needed for the first layer

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# _______________________Plot Training Results_______________________
def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    # Path to the dataset
    dataset_path =  "/kaggle/input/input-data/isolated_words_per_user"

    # Load the dataset
    print("Loading dataset...")
    img_size = (128, 128)
    data, labels = load_dataset(dataset_path, img_size)

    if data.size == 0:
        print("Failed to load dataset. Exiting...")
        exit()

    print(f"Dataset loaded successfully! Number of samples: {len(data)}, Number of users: {len(set(labels))}")

    # Preprocess data
    data = data / 255.0
    data = np.expand_dims(data, axis=-1)

    label_map = {label: idx for idx, label in enumerate(set(labels))}
    encoded_labels = np.array([label_map[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(label_map))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(label_map))

    input_shape = (img_size[0], img_size[1], 1)
    num_classes = len(label_map)

    # Hyperparameter configurations
    configs = [
        {"filters": [32, 64], "dense_units": 128, "dropout_rate": 0.5, "epochs": 20, "batch_size": 32},
        {"filters": [64, 128], "dense_units": 256, "dropout_rate": 0.4, "epochs": 25, "batch_size": 64},
        {"filters": [32, 64, 128], "dense_units": 128, "dropout_rate": 0.3, "epochs": 40, "batch_size": 64},
        {"filters": [32, 64, 128, 256], "dense_units": 256, "dropout_rate": 0.3, "epochs": 40, "batch_size": 64},
        {"filters": [32, 64, 128, 256, 512], "dense_units": 512, "dropout_rate": 0.2, "epochs": 40, "batch_size": 64},
    ]

    best_model = None
    best_val_accuracy = 0
    # Track the best configuration
    best_config = None
    
    for i, config in enumerate(configs):
        print(f"\nTraining model {i + 1} with config: {config}")
        model = build_custom_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=config["filters"],
            dense_units=config["dense_units"],
            dropout_rate=config["dropout_rate"],
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
        )

        plot_training_history(history, title=f"Model {i + 1} - Filters: {config['filters']}, Dense Units: {config['dense_units']}")

        val_accuracy = max(history.history['val_accuracy'])
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            best_config = config  # Save the best configuration

    # Print the results
    print(f"\nBest model achieved validation accuracy of {best_val_accuracy:.4f}")
    print(f"Best model configuration: {best_config}")

    # Final Evaluation on Test Data
    loss, accuracy = best_model.evaluate(X_test, y_test, batch_size=64)
    print(f'Final Test Loss: {loss * 100:.2f} %')
    print(f'Final Test Accuracy: {accuracy * 100:.2f} %')

    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=5,  # Rotate images randomly up to 5 degrees
        width_shift_range=0.1,  # Shift images horizontally by up to 10% of the width
        zoom_range=0.2,  # Zoom in by up to 20%
        fill_mode='nearest'  # Fill in newly created pixels with the nearest pixel value
    )

    # Train the model with data augmentation
    print("Training the model with data augmentation...")
    history_aug = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),  # Data augmentation during training
        steps_per_epoch=len(X_train) // 64,
        epochs=40,
        validation_data=(X_test, y_test)
    )

    # Plot training history
    plot_training_history(history_aug, title="Model Training with Data Augmentation")

    # Compare accuracy before and after augmentation
    original_accuracy = best_val_accuracy
    augmented_accuracy = max(history_aug.history['val_accuracy'])

    print(f"\nValidation accuracy before augmentation: {original_accuracy:.4f}")
    print(f"Validation accuracy after augmentation: {augmented_accuracy:.4f}")
    