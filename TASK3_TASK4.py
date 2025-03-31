# Shiyar Mohammad - 1210766
# Jana Qutosa - 1210331
import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50  
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet152V2

#the base directory where the user's data is stored
base_dir = '/kaggle/input/input-data/isolated_words_per_user'
output_dir = '/kaggle/output/split_data'

#paths for the training, validation, and testing directories
train_dir = os.path.join(output_dir, 'training')
val_dir = os.path.join(output_dir, 'validation')
test_dir = os.path.join(output_dir, 'testing')

# Create the output directories 
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over each user's folder in the base directory
for user_folder in os.listdir(base_dir):
    user_path = os.path.join(base_dir, user_folder)
    if os.path.isdir(user_path):  # Ensure it is a directory
        images = os.listdir(user_path)  # List all files in the user's directory
        
        # Filter the list to include only image files
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split the images into training, validation, and testing sets
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)  # 20% for testing
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=42)  # 25% of training for validation
        
        # Create subdirectories for the user in each data split folder
        os.makedirs(os.path.join(train_dir, user_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, user_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, user_folder), exist_ok=True)
        
        # Copy training images to the training directory
        for img in train_imgs:
            shutil.copy(os.path.join(user_path, img), os.path.join(train_dir, user_folder, img))
        
        # Copy validation images to the validation directory
        for img in val_imgs:
            shutil.copy(os.path.join(user_path, img), os.path.join(val_dir, user_folder, img))
        
        # Copy testing images to the testing directory
        for img in test_imgs:
            shutil.copy(os.path.join(user_path, img), os.path.join(test_dir, user_folder, img))

print("Data splitted successfully.")

#____________________TASK3___________________________________

import os  
import shutil 
import torch  
import torchvision.transforms as transforms  
from torchvision.datasets import ImageFolder  
from torch.utils.data import DataLoader, ConcatDataset  
from torchvision import models  
import torch.nn as nn 
import torch.optim as optim  
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt  

# Define a common resize dimension for all images
resize_dim = (224, 224)

# Define transformations for data augmentation
transformations = {
    'original': transforms.Compose([  # No augmentation; just resize and convert to tensor
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
    ]),
    'rotate': transforms.Compose([  # Rotate images randomly by 45 degrees
        transforms.Resize(resize_dim),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
    ]),
    'scale': transforms.Compose([  # Resize without additional augmentation
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
    ]),
    'color_jitter': transforms.Compose([  # Adjust brightness, contrast, saturation, and hue
        transforms.Resize(resize_dim),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
    ])
}

# Load the original dataset with the 'original' transformation
original_dataset = ImageFolder(root='/kaggle/input/input-data/isolated_words_per_user', transform=transformations['original'])

# Create augmented datasets using other transformations
augmented_datasets = {name: ImageFolder(root='/kaggle/input/input-data/isolated_words_per_user', transform=transform)
                      for name, transform in transformations.items() if name != 'original'}

# Combine the original and augmented datasets
combined_dataset = ConcatDataset([original_dataset] + list(augmented_datasets.values()))

# Split the combined dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(combined_dataset))  # Calculate training size
val_size = len(combined_dataset) - train_size  # Calculate validation size
train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])  # Split datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # DataLoader for training
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # DataLoader for validation

# Load a pre-trained ResNet18 model
model = models.resnet18(weights=None)  # Use untrained ResNet18 architecture
num_features = model.fc.in_features  # Get the number of features from the fully connected layer
model.fc = nn.Linear(num_features, len(original_dataset.classes))  # Replace the fully connected layer for classification

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001

# Training setup
num_epochs = 10  # Number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
model = model.to(device)  # Move the model to the selected device

# Initialize lists to store metrics for plotting
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

# Training and validation loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0  # Accumulate training loss
    correct_train = 0  # Count correct predictions
    total_train = 0  # Count total samples
    for images, labels in train_loader:  # Iterate over training batches
        images, labels = images.to(device), labels.to(device)  # Move data to the device
        
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        train_loss += loss.item() * images.size(0)  # Accumulate loss
        _, predicted = outputs.max(1)  # Get predicted class
        correct_train += predicted.eq(labels).sum().item()  # Count correct predictions
        total_train += labels.size(0)  # Count total samples
    
    # Calculate average training loss and accuracy
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = 100.0 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0  # Accumulate validation loss
    correct_val = 0  # Count correct predictions
    total_val = 0  # Count total samples
    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:  # Iterate over validation batches
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item() * images.size(0)  # Accumulate loss
            
            _, predicted = outputs.max(1)  # Get predicted class
            correct_val += predicted.eq(labels).sum().item()  # Count correct predictions
            total_val += labels.size(0)  # Count total samples
    
    # Calculate average validation loss and accuracy
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100.0 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Print metrics for the current epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
          f'Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

# Plot accuracy and loss
plt.figure(figsize=(14, 6))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()  

#__________________________TASK4________________________________________


# Directories for the dataset that you uploaded
train_dir = '/kaggle/output/split_data/training'
val_dir = '/kaggle/output/split_data/validation'
test_dir = '/kaggle/output/split_data/testing'

# Set Up Data Augmentation for Training and Validation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rescale=1.0/255
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load Pre-trained ResNet152V2 Model for Transfer Learning
base_model = ResNet152V2(weights='/kaggle/input/resnet152v2/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                         include_top=False, 
                         input_shape=(224, 224, 3))

# Freeze the first 100 layers of the base model
for layer in base_model.layers[:100]:
    layer.trainable = False

# Allow the rest of the layers to be trainable
for layer in base_model.layers[100:]:
    layer.trainable = True

# Step 3: Build Custom Model on Top of Pre-trained ResNet152V2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Regularization with Dropout
num_classes = train_generator.num_classes  # Number of classes in your dataset
x = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set Up Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the Model
history = model.fit(
    train_generator,
    epochs=50,  
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the Fine-Tuned Model on Test Data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"Fine-Tuned Test Accuracy: {test_acc * 100:.2f}%")

# Plot training vs validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training vs validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

