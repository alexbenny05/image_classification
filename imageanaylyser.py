import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# -------------------------------
# CIFAR-10 MANUAL LOADER
# -------------------------------

def load_cifar_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    data = batch[b'data']
    labels = batch[b'labels']

    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)  # CHW -> HWC
    return data, np.array(labels)


def load_cifar10(dataset_path):
    X_train, y_train = [], []

    for i in range(1, 6):
        data, labels = load_cifar_batch(
            os.path.join(dataset_path, f'data_batch_{i}')
        )
        X_train.append(data)
        y_train.append(labels)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_cifar_batch(
        os.path.join(dataset_path, 'test_batch')
    )

    return X_train, y_train, X_test, y_test


# -------------------------------
# LOAD DATASET
# -------------------------------

dataset_path = "cifar-10-batches-py"

X_train, y_train, X_test, y_test = load_cifar10(dataset_path)

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

print("Dataset loaded successfully!")
print("Total images:", X.shape[0])

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    normalized = resized / 255.0
    return normalized

X_processed = np.array([preprocess_image(img) for img in X])

# Flatten images for ML
X_flattened = X_processed.reshape(X_processed.shape[0], -1)

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_flattened, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL TRAINING (SVM)
# -------------------------------

model = SVC(kernel='linear')
model.fit(X_train, y_train)

print("Model training completed!")

# -------------------------------
# PREDICTION & EVALUATION
# -------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", cm)

# -------------------------------
# IMAGE ENHANCEMENT DEMO
# -------------------------------

sample = X[0]

bright = cv2.convertScaleAbs(sample, alpha=1.2, beta=30)
blur = cv2.GaussianBlur(sample, (5, 5), 0)

plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(sample)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Brightness Adjusted")
plt.imshow(bright)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Gaussian Blur")
plt.imshow(blur)
plt.axis("off")

plt.show()

# -------------------------------
# VISUALIZE PREDICTIONS
# -------------------------------

labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

plt.figure(figsize=(10, 5))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    img = X_test[i].reshape(32, 32)
    plt.imshow(img, cmap='gray')
    plt.title(f"P:{labels[y_pred[i]]}\nA:{labels[y_test[i]]}")
    plt.axis('off')

plt.show()
