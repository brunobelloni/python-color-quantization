import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torchvision import datasets

import main as functions
from main import sequence


def assign_labels(x, cluster):
    return np.argmin(np.linalg.norm(x[:, None] - cluster, axis=-1), axis=-1)


def main():
    # Load the MNIST training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)

    train_images = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_images = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()
    print(f'Train images shape: {train_images.shape}')
    print(f'Test images shape: {test_images.shape}\n')

    # convert each image to 1 dimensional array
    train_images = train_images.reshape(len(train_images), -1)
    test_images = test_images.reshape(len(test_images), -1)

    train_images = train_images.astype(np.float64) / 255.0
    test_images = test_images.astype(np.float64) / 255.0

    for algorithm in ['bkm', 'ibkm', 'okm', 'iokm']:
        sequence.reset()
        kwargs = {
            'x': train_images,
            'k': 10,  # MNIST dataset has 10 classes
        }
        if algorithm == 'iokm':
            kwargs['lr_exp'] = 0.1
            kwargs['sample_rate'] = 1.0
        cluster, _ = getattr(functions, algorithm)(**kwargs)

        # Assign each pixel to the nearest cluster centroid
        pred_train_labels = assign_labels(train_images, cluster)
        pred_test_labels = assign_labels(test_images, cluster)

        train_acc = accuracy_score(y_true=train_labels, y_pred=pred_train_labels)
        train_recall = recall_score(y_true=train_labels, y_pred=pred_train_labels, average='macro')
        train_f1 = f1_score(y_true=train_labels, y_pred=pred_train_labels, average='macro')
        test_acc = accuracy_score(y_true=test_labels, y_pred=pred_test_labels)
        test_recall = recall_score(y_true=test_labels, y_pred=pred_test_labels, average='macro')
        test_f1 = f1_score(y_true=test_labels, y_pred=pred_test_labels, average='macro')

        print(
            f"{algorithm}: Train Acc: {train_acc:.2f}; Train Recall: {train_recall:.2f}; Train F1: {train_f1:.2f}\n"
            f"{algorithm}: Test Acc: {test_acc:.2f}; Test Recall: {test_recall:.2f}; Test F1: {test_f1:.2f}\n"
        )


if __name__ == '__main__':
    main()
