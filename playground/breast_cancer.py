import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import main as functions
from playground.mnist import assign_labels


def main():
    # Load the Breast Cancer Wisconsin Diagnostic dataset
    cancer_dataset = load_breast_cancer()

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        cancer_dataset.data,
        cancer_dataset.target,
        test_size=0.2,
        random_state=42,
    )

    print(f"Train data shape: {x_train.shape}")
    print(f"Test data shapea: {x_test.shape}")

    for algorithm in ['bkm', 'ibkm', 'okm', 'iokm']:
        kwargs = {
            'x': x_train,
            'k': 2,  # Breast cancer dataset has 2 classes
            'image': np.zeros((21, 21, 3)),  # okm/iokm
        }
        if algorithm == 'iokm':
            kwargs['lr_exp'] = 0.1
            kwargs['sample_rate'] = 1.0
        cluster, _, _ = getattr(functions, algorithm)(**kwargs)

        # Assign each instance to the nearest cluster centroid
        pred_train_labels = assign_labels(x_train, cluster)
        pred_test_labels = assign_labels(x_test, cluster)

        train_acc = accuracy_score(y_true=y_train, y_pred=pred_train_labels)
        train_recall = recall_score(y_true=y_train, y_pred=pred_train_labels, average='macro')
        train_f1 = f1_score(y_true=y_train, y_pred=pred_train_labels, average='macro')
        test_acc = accuracy_score(y_true=y_test, y_pred=pred_test_labels)
        test_recall = recall_score(y_true=y_test, y_pred=pred_test_labels, average='macro')
        test_f1 = f1_score(y_true=y_test, y_pred=pred_test_labels, average='macro')

        print(
            f"{algorithm}: Train Acc: {train_acc:.2f}; Train Recall: {train_recall:.2f}; Train F1: {train_f1:.2f}\n"
            f"{algorithm}: Test Acc: {test_acc:.2f}; Test Recall: {test_recall:.2f}; Test F1: {test_f1:.2f}\n"
        )


if __name__ == '__main__':
    main()
