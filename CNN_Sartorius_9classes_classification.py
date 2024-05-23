import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
train_directory = 'D:\\Data for machine learning\\Sartorius cell instance segmentation\\LIVECell_dataset_2021\\images\\livecell_train_val_images'
test_directory = 'D:\\Data for machine learning\\Sartorius cell instance segmentation\\LIVECell_dataset_2021\\images\\livecell_test_images'

# Define target size for resizing images
target_size = (128, 128)

# Function to load images and their labels from a directory
def load_images(directory, img_dataset, img_labels):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".tif"):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, target_size)
                label = os.path.basename(os.path.dirname(image_path))
                img_dataset.append(image.astype('float32'))
                img_labels.append(label)

# Load training images
img_dataset_train, img_labels_train = [], []
load_images(train_directory, img_dataset_train, img_labels_train)
X_train, L_train = np.asarray(img_dataset_train), np.asarray(img_labels_train)

# Load testing images
img_dataset_test, img_labels_test = [], []
load_images(test_directory, img_dataset_test, img_labels_test)
X_test, L_test = np.asarray(img_dataset_test), np.asarray(img_labels_test)

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Define class labels and create a mapping
class_labels = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'RatC6', 'SHSY5Y', 'SkBr3', 'SKOV3']
label_map = {class_label: i for i, class_label in enumerate(class_labels)}

# Convert string labels to numeric and one-hot encode them
L_train_numeric = np.vectorize(label_map.get)(L_train)
L_train = to_categorical(L_train_numeric, num_classes=len(class_labels))
L_test_numeric = np.vectorize(label_map.get)(L_test)
L_test = to_categorical(L_test_numeric, num_classes=len(class_labels))

# Reshape and normalize the image data based on the backend image format
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    input_shape = (1, X_train.shape[1], X_train.shape[2])
else:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Split the training data into training and validation sets
X_train, X_val, L_train, L_val = train_test_split(X_train, L_train, test_size=0.05, random_state=2)

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Create augmented data
X_train_augmented = []
L_train_augmented = []

for x, y in datagen.flow(X_train, L_train, batch_size=len(X_train), shuffle=False):
    X_train_augmented.append(x)
    L_train_augmented.append(y)
    if len(X_train_augmented) * len(x) >= len(X_train):
        break

X_train_augmented = np.concatenate(X_train_augmented)
L_train_augmented = np.concatenate(L_train_augmented)

# Concatenate original and augmented data
X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
L_train_combined = np.concatenate((L_train, L_train_augmented), axis=0)

# Shuffle the combined dataset
shuffled_indices = np.random.permutation(len(X_train_combined))
X_train_combined = X_train_combined[shuffled_indices]
L_train_combined = L_train_combined[shuffled_indices]

print('Combined training images shape:', X_train_combined.shape)
print('Combined training labels shape:', L_train_combined.shape)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Define callbacks for early stopping and learning rate scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr *= 0.9
    return lr

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
history = model.fit(X_train_combined, L_train_combined, epochs=100, batch_size=32, validation_data=(X_val, L_val), callbacks=[early_stopping, lr_callback])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, L_test, batch_size=64)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot loss
loss_path = os.path.join(SCRIPT_DIR, 'loss_plot.png')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(loss_path, dpi=300, facecolor='white')

# Plot accuracy
accuracy_path = os.path.join(SCRIPT_DIR, 'accuracy_plot.png')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(accuracy_path, dpi=300, facecolor='white')

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title=None, xlabel='Predicted Label', ylabel='True Label', cmap='Purples'):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()

# Predict classes for test data
predictions = model.predict(X_test)
y_true = np.argmax(L_test, axis=1)
y_pred = np.argmax(predictions, axis=1)

# Paths for saving plots
confusion_path_without_norm = os.path.join(SCRIPT_DIR, 'confusion_matrix_without_normalization.png')
confusion_path_norm = os.path.join(SCRIPT_DIR, 'normalized_confusion_matrix.png')

# Plot confusion matrices with/without normalizations and save them
plot_confusion_matrix(y_true, y_pred, class_labels, normalize=False, title=f'Confusion Matrix without Normalization\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}', cmap='Greens')
plt.savefig(confusion_path_without_norm, dpi=300, facecolor='white')

plot_confusion_matrix(y_true, y_pred, class_labels, normalize=True, title=f'Normalized Confusion Matrix\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}', cmap='Purples')
plt.savefig(confusion_path_norm, dpi=300, facecolor='white')

# Save the model
model_json_path = os.path.join(SCRIPT_DIR, "model.json")
model_weights_path = os.path.join(SCRIPT_DIR, "model_weights.h5")
with open(model_json_path, "w") as json_file:
    json_file.write(model.to_json(indent=4))
model.save_weights(model_weights_path)
print("Model saved successfully!")