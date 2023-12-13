import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Define paths
train_data_dir = 'chest_xray_prediction\chest_xray\\train'
validation_data_dir = 'chest_xray_prediction\chest_xray\\val'
test_data_dir = 'chest_xray_prediction\chest_xray\\test'






# Parameters
img_width, img_height = 224, 224
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation and test data should not be augmented
val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Set shuffle to False for evaluation
)

# Calculate steps per epoch dynamically
steps_per_epoch = train_generator.samples // batch_size
if train_generator.samples % batch_size != 0:
    steps_per_epoch += 1

# Calculate validation and test steps dynamically
validation_steps = validation_generator.samples // batch_size
if validation_generator.samples % batch_size != 0:
    validation_steps += 1

test_steps = test_generator.samples // batch_size
if test_generator.samples % batch_size != 0:
    test_steps += 1

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
model_checkpoint = ModelCheckpoint('model/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tensorboard = TensorBoard(log_dir='logs')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-6)

callbacks = [model_checkpoint, early_stopping, tensorboard, reduce_lr]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Save the trained model
model.save('model/your_trained_model.h5')

# Confusion Matrix and Classification Report for the test set
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, steps=test_steps)
y_pred = np.round(y_pred_probs)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
plot_confusion_matrix(conf_mat=cm, show_normed=True, class_names=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix - Test Set')
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_true, y_pred))
