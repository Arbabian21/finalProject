import numpy as np

# Define the three confusion matrices
confusion_matrices = {
    "Random Forest": np.array([
        [7825, 352, 357, 22],  # nonDRNA
        [30, 171, 11, 0],      # RNA
        [4, 0, 23, 0],         # DNA
        [0, 0, 0, 0]           # DRNA
    ]),
    "Gradient Boosting Trees": np.array([
        [7810, 330, 347, 22],  # nonDRNA
        [36, 186, 15, 0],      # RNA
        [13, 7, 29, 0],        # DNA
        [0, 0, 0, 0]           # DRNA
    ]),
    "k-Nearest Neighbors": np.array([
        [7829, 384, 362, 22],  # nonDRNA
        [23, 152, 9, 0],       # RNA
        [7, 7, 20, 0],         # DNA
        [0, 0, 0, 0]           # DRNA
    ])
}

# Labels
labels = ['nonDRNA', 'RNA', 'DNA', 'DRNA']

# Function to calculate metrics
def calculate_metrics(conf_matrix):
    metrics = {}
    for i, label in enumerate(labels):
        TP = conf_matrix[i, i]
        FN = np.sum(conf_matrix[i, :]) - TP
        FP = np.sum(conf_matrix[:, i]) - TP
        TN = np.sum(conf_matrix) - (TP + FP + FN)

        sensitivity = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = 100 * TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = 100 * (TP + TN) / (TP + FP + TN + FN)
        mcc_numerator = (TP * TN) - (FP * FN)
        mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

        metrics[label] = {
            'Sensitivity': round(sensitivity, 1),
            'Specificity': round(specificity, 1),
            'Accuracy': round(accuracy, 1),
            'MCC': round(mcc, 3)
        }

    return metrics

# Calculate average MCC and accuracy4labels// 
def calculate_overall_metrics(metrics, conf_matrix):
    mcc_values = [metrics[label]['MCC'] for label in labels]
    average_mcc = round(sum(mcc_values) / len(mcc_values), 3)
    TP_all = sum(conf_matrix[i, i] for i in range(len(labels)))
    accuracy_4_labels = round(100 * TP_all / np.sum(conf_matrix), 1)

    return average_mcc, accuracy_4_labels

# Process each confusion matrix
for matrix_name, conf_matrix in confusion_matrices.items():
    print(f"Results for {matrix_name}:\n")
    
    # Compute metrics for each label
    metrics = calculate_metrics(conf_matrix)
    
    # Compute average MCC and accuracy4labels
    average_mcc, accuracy_4_labels = calculate_overall_metrics(metrics, conf_matrix)
    
    # Display results
    for label, metric in metrics.items():
        print(f"{label}: {metric}")
    print(f"\nAverage MCC: {average_mcc}")
    print(f"Accuracy (4 labels): {accuracy_4_labels}%\n")
    print("-" * 50)
