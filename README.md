

# 🐟 Fish Disease Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify fish as either **Infected** or **Fresh (Healthy)** based on images. It involves image preprocessing, training a custom CNN model, and evaluating its performance on a real-world dataset.

---

## 📁 Dataset

* Dataset: Fish Disease Dataset (compressed as `fish_disease.zip`)
* Classes:

  * `InfectedFish`
  * `FreshFish`
* Format: JPEG/PNG images organized in subfolders by class.

---

## 📦 Project Structure

```
fish_disease_project/
│
├── fishdd.ipynb          # Jupyter Notebook with full pipeline
├── NTD_1.h5              # Trained CNN model
├── README.md             # This file
└── dataset/
    ├── InfectedFish/
    └── FreshFish/
```

---

## 🔧 Steps and Code Breakdown

### 1. Dataset Extraction

```python
os.rename("/content/archive (1).zip", "/content/fish_disease.zip")
with zipfile.ZipFile("/content/fish_disease.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/fish_disease_data")
```

✅ Renames and extracts the dataset into a working directory.

---

### 2. Loading Class Names

```python
def find_Class(directory_path):
    ...
```

✅ Returns all class folder names (e.g., `InfectedFish`, `FreshFish`).

---

### 3. Image Preprocessing

```python
img_arr = cv2.imread(img_path)
img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
img_arr = cv2.resize(img_arr, (128, 128))
```

* All images are resized to 128×128 pixels.
* RGB conversion is applied.
* Normalized pixel values to the \[0, 1] range.
* Final data is split into `X` (features) and `Y` (labels).

---

### 4. Data Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
```

* 90% data used for training.
* 10% held out for testing.

---

### 5. Model Architecture

```python
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
...
model.add(Dense(2, activation='softmax'))
```

* 3 Convolutional Layers with ReLU + MaxPooling
* Flatten layer → Dense layer with 2 outputs (Softmax)
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy

---

### 6. Training

```python
model.fit(X_train, y_train, epochs=50, validation_split=0.15)
```

* Trained for 50 epochs
* Validation split: 15% of training data

📊 Accuracy and loss plots are generated to visualize overfitting/underfitting.

---

### 7. Evaluation

```python
model.evaluate(X_test, y_test)
```

* Final test accuracy: **96.77%**
* Classification report:

```
              precision    recall  f1-score   support

 InfectedFish     0.93       1.00      0.97        14
    FreshFish     1.00       0.94      0.97        17

    Accuracy                         0.97        31
```

---
![image](https://github.com/user-attachments/assets/0a803ad7-b119-4d47-974a-89ed52f33043)
![image](https://github.com/user-attachments/assets/c9104f6f-fbf8-432f-89ee-aa80c6cab7e8)
![image](https://github.com/user-attachments/assets/347da924-013c-479b-a35d-d648a7ad4f4b)

### 8. Inference on New Image

```python
img = image.load_img("path_to_img", target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.0
prediction = model.predict(img)
```

🔍 Predicts whether a new image is **Infected** or **Fresh**, and prints class name + confidence score.

---

## ✅ Final Results

* 🎯 Test Accuracy: 96.77%
* ✔️ Perfect recall on infected class
* ✔️ Balanced performance across both classes

---

## 🚀 Future Improvements

* Add dropout to prevent overfitting
* Try deeper architectures like VGG or ResNet
* Use data augmentation to increase robustness
* Deploy as a Streamlit web app

---

## 📌 Dependencies

* Python 3.8+
* TensorFlow / Keras
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib

You can install all dependencies using:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib
```

