import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# DATASET PATH
# ==============================
dataset_path ="train"  

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 64
MAX_IMAGES = 2000

images = []
labels = []

# ==============================
# CHECK DATASET
# ==============================
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    print("Current directory:", os.getcwd())
    exit()

print("Loading images...")

count = 0

for filename in os.listdir(dataset_path):

    if count >= MAX_IMAGES:
        break

    if filename.startswith("cat"):

        label = 0

    elif filename.startswith("dog"):

        label = 1

    else:
        continue

    img_path = os.path.join(dataset_path, filename)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    images.append(img.flatten())
    labels.append(label)

    count += 1

print("Images loaded:", len(images))

# ==============================
# NUMPY ARRAYS
# ==============================
X = np.array(images)
y = np.array(labels)

print("Dataset shape:", X.shape)

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# TRAIN SVM
# ==============================
print("\nTraining SVM...")

svm = SVC(
    kernel="linear",
    C=1.0
)

svm.fit(X_train, y_train)

print("Training completed!")

# ==============================
# PREDICTIONS
# ==============================
y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# DISPLAY PREDICTIONS
# ==============================
plt.figure(figsize=(12, 4))

num_images = min(5, len(X_test))

for i in range(num_images):

    plt.subplot(1, num_images, i + 1)

    image = X_test[i].reshape(IMG_SIZE, IMG_SIZE)

    predicted_label = "Cat" if y_pred[i] == 0 else "Dog"

    plt.imshow(image, cmap="gray")
    plt.title(predicted_label)
    plt.axis("off")

plt.tight_layout()
plt.show()