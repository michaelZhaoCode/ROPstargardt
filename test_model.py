import caffe
import numpy as np
import lmdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Ensure correct command-line usage
if len(sys.argv) < 5:
    print("Usage: python test_caffe.py <deploy.prototxt> <model.caffemodel> <test_lmdb> <activation_type>")
    print("activation_type: 'softmax' or 'sigmoid'")
    sys.exit(1)

# Get command-line arguments
deploy_prototxt = sys.argv[1]   # Path to deploy.prototxt
caffe_model = sys.argv[2]       # Path to trained .caffemodel
test_lmdb = sys.argv[3]         # Path to LMDB test dataset
activation_type = sys.argv[4].lower()  # Activation type: 'softmax' or 'sigmoid'

if activation_type not in ["softmax", "sigmoid"]:
    print("Error: activation_type must be either 'softmax' or 'sigmoid'.")
    sys.exit(1)

# Set Caffe to GPU mode
caffe.set_mode_gpu()
caffe.set_devices([0, 1])  # Select GPU IDs

# Load the network
net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)

# Open LMDB test dataset
lmdb_env = lmdb.open(test_lmdb, readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

# Variables to compute overall accuracy
total_samples = 0
correct_predictions = 0

# Lists to store per-sample predictions for further metrics
all_true_labels = []
all_predicted_labels = []
all_probabilities = []

# Confusion matrix counters
tp = 0  # True positives
tn = 0  # True negatives
fp = 0  # False positives
fn = 0  # False negatives

# Determine batch size from the network
batch_size = net.blobs['data'].data.shape[0]
batch_data = []
batch_labels = []

# Iterate through the LMDB dataset
for key, value in lmdb_cursor:
    total_samples += 1
    datum = caffe_pb2.Datum()
    datum.ParseFromString(value)

    # Convert LMDB data to image array
    img = np.frombuffer(datum.data, dtype=np.uint8)
    img = img.reshape(datum.channels, datum.height, datum.width)

    # Preprocess the image to match training (normalize, crop, etc.)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = img[:, :128, :128]  # Crop if necessary

    # Append to batch
    batch_data.append(img)
    batch_labels.append(int(datum.label))  # Ensure labels are integers

    # Process the batch when full
    if len(batch_data) == batch_size:
        # Perform batch inference
        net.blobs['data'].data[...] = np.array(batch_data)
        output = net.forward()

        # Process predictions based on activation type
        if activation_type == "sigmoid":
            # Get sigmoid probabilities (1D array, one probability per sample)
            probs = output['sigmoid'].flatten()
            predictions = (probs > 0.5).astype(int)
        else:  # Softmax case
            # Get softmax probabilities; expected shape: (batch_size, 2)
            probs = output['prob']
            predictions = np.argmax(probs, axis=1)

        # Process each sample in the batch
        for i in range(batch_size):
            true_label = batch_labels[i]
            if activation_type == "sigmoid":
                prob = probs[i]
            else:
                # For softmax, take the probability for the positive class (index 1)
                prob = probs[i][1]

            pred_label = predictions[i]

            # Update overall accuracy count
            if pred_label == true_label:
                correct_predictions += 1

            # Update confusion matrix counters
            if true_label == 1 and pred_label == 1:
                tp += 1
            elif true_label == 0 and pred_label == 0:
                tn += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 1 and pred_label == 0:
                fn += 1

            # Save metrics for later evaluation
            all_true_labels.append(true_label)
            all_predicted_labels.append(pred_label)
            all_probabilities.append(prob)

        # Clear batch data for the next batch
        batch_data.clear()
        batch_labels.clear()

# Final overall accuracy computation
accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

# Compute additional metrics using scikit-learn
precision = precision_score(all_true_labels, all_predicted_labels)
recall = recall_score(all_true_labels, all_predicted_labels)  # Sensitivity (TPR)
f1 = f1_score(all_true_labels, all_predicted_labels)

auc = roc_auc_score(all_true_labels, all_probabilities)

# Calculate specificity: TN / (TN + FP)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Define labels in the desired order: class 1 first, then class 0
labels = [1, 0]

# Compute the confusion matrix with specified label order
cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=labels)

# Print evaluation metrics
print(f"\nTotal Samples Evaluated: {total_samples}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {auc:.4f}")
# Create a DataFrame with labeled rows and columns
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels]
)

print("Confusion Matrix:")
print(cm_df)

# --------------------------------------------------------------------
# Assert statements to confirm the consistency of the metrics

# 1. Total samples consistency: the sum of confusion matrix counters must equal total_samples.
assert (tp + tn + fp + fn) == total_samples, "Mismatch in total sample count vs. confusion matrix sum."

# 2. Accuracy consistency: Accuracy should equal (TP + TN) / total_samples * 100.
calculated_accuracy = (tp + tn) / total_samples * 100.0
assert abs(accuracy - calculated_accuracy) < 1e-4, f"Accuracy mismatch: {accuracy} vs {calculated_accuracy}"

# 3. Precision consistency: when positive predictions exist, precision should equal TP / (TP + FP).
if (tp + fp) > 0:
    calculated_precision = tp / (tp + fp)
    assert abs(precision - calculated_precision) < 1e-4, f"Precision mismatch: {precision} vs {calculated_precision}"

# 4. Recall consistency: recall should equal TP / (TP + FN).
if (tp + fn) > 0:
    calculated_recall = tp / (tp + fn)
    assert abs(recall - calculated_recall) < 1e-4, f"Recall mismatch: {recall} vs {calculated_recall}"

# 5. F1 Score consistency: F1 should equal 2 * (precision * recall) / (precision + recall).
if (precision + recall) > 0:
    calculated_f1 = 2 * (precision * recall) / (precision + recall)
    assert abs(f1 - calculated_f1) < 1e-4, f"F1 mismatch: {f1} vs {calculated_f1}"

# 6. Specificity consistency: specificity should equal TN / (TN + FP).
if (tn + fp) > 0:
    calculated_specificity = tn / (tn + fp)
    assert abs(specificity - calculated_specificity) < 1e-4, f"Specificity mismatch: {specificity} vs {calculated_specificity}"

print("All metric consistency checks passed.")

# For softmax, use Youden's index to determine the optimal threshold
if activation_type == "softmax":
    # Ensure you are using the probability of the positive class
    all_probabilities = np.array(all_probabilities)
    if all_probabilities.ndim == 2 and all_probabilities.shape[1] == 2:
        all_probabilities = all_probabilities[:, 1]  # Positive class probability

    # Compute ROC curve (FPR, TPR, thresholds)
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_probabilities)

    # Calculate Youden's index for each threshold: (TPR - FPR)
    youden_index = tpr - fpr

    # Select the threshold that maximizes Youden's index
    best_threshold = thresholds[np.argmax(youden_index)]
    print(f"Max Youden's Index: {np.max(youden_index):.4f}")
    print(f"\nBest threshold based on Youden's index: {best_threshold:.4f}")

    # Generate new predictions using the best threshold
    optimal_predicted_labels = (all_probabilities >= best_threshold).astype(int)

    # Compute optimal evaluation metrics
    optimal_accuracy = np.mean(optimal_predicted_labels == np.array(all_true_labels)) * 100
    optimal_cm = confusion_matrix(all_true_labels, optimal_predicted_labels)
    tn, fp, fn, tp = optimal_cm.ravel()

    optimal_precision = precision_score(all_true_labels, optimal_predicted_labels, zero_division=0)
    optimal_recall = recall_score(all_true_labels, optimal_predicted_labels)
    optimal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    optimal_f1 = f1_score(all_true_labels, optimal_predicted_labels)
    optimal_auc = roc_auc_score(all_true_labels, all_probabilities)

    # Display optimal threshold evaluation
    print(f"Accuracy using optimal threshold: {optimal_accuracy:.2f}%")
    print(f"Precision: {optimal_precision:.4f}")
    print(f"Sensitivity (Recall): {optimal_recall:.4f}")
    print(f"Specificity: {optimal_specificity:.4f}")
    print(f"F1 Score: {optimal_f1:.4f}")
    print(f"ROC AUC: {optimal_auc:.4f}")
    print("Confusion Matrix with optimal threshold:")

    reordered_cm = np.array([
        [optimal_cm[1,1], optimal_cm[1,0]],  # Actual 1: [TP, FN]
        [optimal_cm[0,1], optimal_cm[0,0]]   # Actual 0: [FP, TN]
    ])

    # Format the confusion matrix for printing with class 1 first
    labels = [1, 0]  # Desired order: class 1, then class 0
    optimal_cm_df = pd.DataFrame(
        reordered_cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )
    print(optimal_cm_df)

# Close LMDB environment
lmdb_env.close()
