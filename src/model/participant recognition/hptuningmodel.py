import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import optuna

PREPROCESSEDFOLDER = "D:/Individual Project/Preprocessed Participant Data"

# Load Data  --------------------------------------------------------------
DATASET_FILE = "dataset1to6 Normalized R2&R3 + DA.npy"
LABEL_FILE = "dataset1to6 Label R2&R3 + DA.npy"

# Load data and labels
data = np.load(f"{PREPROCESSEDFOLDER}/{DATASET_FILE}")
labels = np.load(f"{PREPROCESSEDFOLDER}/{LABEL_FILE}")

# Shuffle data and labels
indices = np.arange(len(data))
np.random.shuffle(indices)

shuffled_data = data[indices]
shuffled_labels = labels[indices]

# Define ratio for splitting up data and labels
train_ratio = 0.75

# Calculate the split point
split_point = int(len(data) * train_ratio)

# Split the data and labels into training and validate set
x_train, x_validate = shuffled_data[:split_point], shuffled_data[split_point:]
y_train, y_validate = shuffled_labels[:split_point], shuffled_labels[split_point:]

print("x_train:", x_train.shape)
print("x_validate:", x_validate.shape)
print("y_train:", y_train.shape)
print("y_validate:", y_validate.shape)
# ----------------------------------------------------------------------------

# Process/Modify Loaded Data -------------------------------------------------
# Save image parameters to the constants that we will use later for data re-shaping and for model traning.
(_, WIDTH, HEIGHT) = x_train.shape
CHANNELS = 1
print('WIDTH:', WIDTH)
print('HEIGHT:', HEIGHT)
print('CHANNELS:', CHANNELS)

# Reshape train datasets
x_train = x_train.reshape(
    x_train.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)
x_validate = x_validate.reshape(
    x_validate.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)
print('x_train:', x_train.shape)
print('x_validate:', x_validate.shape)

# Process labels to have correct shape (x, 6)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=61)
y_validate = tf.keras.utils.to_categorical(y_validate, num_classes=61)
# ----------------------------------------------------------------------------

# Hyperparameters Optimization -----------------------------------------------
def create_model(trial):
    global WIDTH, HEIGHT, CHANNELS
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv1_filters", 32, 256), (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv2_filters", 32, 256), (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv3_filters", 32, 256), (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(trial.suggest_int("conv4_filters", 32, 256), (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(trial.suggest_int("dense_units1", 64, 256), activation='relu'))
    model.add(tf.keras.layers.Dropout(trial.suggest_float("dropout1", 0.1, 0.5)))
    model.add(tf.keras.layers.Dense(trial.suggest_int("dense_units2", 64, 128), activation='relu'))
    model.add(tf.keras.layers.Dropout(trial.suggest_float("dropout2", 0.1, 0.5)))
    model.add(tf.keras.layers.Dense(61, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("learning_rate", 0.0005, 0.002))
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def objective(trial):
    global x_train, y_train, x_validate, y_validate

    # Create a model with trial-specific hyperparameters
    model = create_model(trial)

    # Train the model
    training_history = model.fit(
        x_train, 
        y_train, 
        epochs=30,
        batch_size=40,
        validation_data=(x_validate, y_validate),
        verbose=0  # Set to 0 to avoid printing progress for each trial
    )

    # Evaluate the model on the validation set
    validation_loss, validation_accuracy = model.evaluate(x_validate, y_validate)

    return validation_loss
# ----------------------------------------------------------------------------

# Define the search space for hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Rebuild and train the model with the best hyperparameters
best_model = create_model(best_params)
training_history = best_model.fit(
    x_train, 
    y_train, 
    epochs=10,
    batch_size=32, 
    validation_data=(x_validate, y_validate)
)

plt.figure()
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='test set')
plt.legend()
plt.show(block=False)

plt.figure()
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
plt.plot(training_history.history['val_accuracy'], label='test set')
plt.legend()
plt.show()

# Evaluate the best model
best_validation_loss, best_validation_accuracy = best_model.evaluate(x_validate, y_validate)
print('Best Validation loss: ', best_validation_loss)
print('Best Validation accuracy: ', best_validation_accuracy)

# Save the best model
model_name = 'best_participant_recognition_cnn.h5'
best_model.save(model_name, save_format='h5')
