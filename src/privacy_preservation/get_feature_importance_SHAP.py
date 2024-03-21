import numpy as np
import tensorflow as tf
import shap
from pathlib import Path

PREPROCESSEDFOLDER = "../../data/processed"

TRAINDATASET = "trainDataset30.npy"
VALIDATIONDATASET = "validationDataset30.npy"
TESTDATASET = "testDataset30.npy"
TRAINED_MODEL = "experimentmodels/multitask_recognition_cnn5_97_70.h5"

script_location = Path(__file__).resolve().parent
data_folder = script_location / PREPROCESSEDFOLDER
train_data_path = data_folder / TRAINDATASET
validation_labels_path = data_folder / VALIDATIONDATASET
test_data_path = data_folder / TESTDATASET

# Load data and labels
# train_data = np.load(f"{PREPROCESSEDFOLDER}/{TRAINDATASET}")
# validate_data = np.load(f"{PREPROCESSEDFOLDER}/{VALIDATIONDATASET}")
# test_data = np.load(f"{PREPROCESSEDFOLDER}/{TESTDATASET}")
train_data = np.load(train_data_path)
validate_data = np.load(validation_labels_path)
test_data = np.load(test_data_path)

x_test = test_data

# Shuffle data and labels
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices)
x_train = train_data[train_indices]

validate_indices = np.arange(len(validate_data))
np.random.shuffle(validate_indices)
x_validate = validate_data[validate_indices]

(_, WIDTH, HEIGHT) = x_test.shape
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

# Reshape validation datasets
x_validate = x_validate.reshape(
    x_validate.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)

# Reshape test datasets
x_test = x_test.reshape(
    x_test.shape[0],
    WIDTH,
    HEIGHT,
    CHANNELS
)

print('x_train:', x_train.shape)
print('x_validate:', x_validate.shape)
print('x_test:', x_test.shape)

# Load the model
model = tf.keras.models.load_model(TRAINED_MODEL)
model_activity = tf.keras.Model(inputs=model.input, outputs=model.output[0])
model_participant = tf.keras.Model(inputs=model.input, outputs=model.output[1])

def compute_shap_values(explainer, data, batch_size=1):
    shap_values_batches = []
    num_samples = data.shape[0]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        data_batch = data[start_idx:end_idx]

        # Compute SHAP values for the current batch
        shap_values_batch = explainer.shap_values(data_batch)

        shap_values_batches.append(shap_values_batch)
        print(f"Processed batch {start_idx//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")

    return shap_values_batches

with tf.device('/cpu:0'):
    print("Start SHAP explainer:")
    # Use train data as the background distribution
    explainer = shap.DeepExplainer(model_participant, x_train[:50])

    print("Compute SHAP values:")
    # Compute SHAP values for a subset of your data
    # shap_values = explainer.shap_values(x_test)
    shap_values = compute_shap_values(explainer, x_test, 1)

# Average the absolute SHAP values across all predictions
mean_shap_values_participant = np.mean(np.abs(shap_values), axis=0)

# Sort features by importance
sorted_feature_indices_participant = np.argsort(mean_shap_values_participant.flatten())[::-1]

# Identify the most important features
print(f"Feature shape: {sorted_feature_indices_participant.shape}")

# Get top k most important features
k = 10000
important_features_participant_topk = sorted_feature_indices_participant[:k]
np.savetxt('important_features_participant10000.txt', important_features_participant_topk, fmt='%.0f')

k = 1000
important_features_participant_topk = sorted_feature_indices_participant[:k]
np.savetxt('important_features_participant1000.txt', important_features_participant_topk, fmt='%.0f')

print("Saved Important Features")