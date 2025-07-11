
# 🧠 Handwritten Digit Classification using MNIST Dataset

A beginner-friendly deep learning project built with TensorFlow and Keras that demonstrates how to classify handwritten digits (0–9) using a simple neural network trained on the MNIST dataset.

---

## 📄 Project Overview
This project performs the following tasks:

✅ Loads and preprocesses the MNIST dataset of handwritten digits.

✅ Defines a simple feed-forward neural network with one hidden layer.

✅ Trains the model on the training set.

✅ Evaluates the trained model on the test set.

✅ Makes predictions and visualizes one sample prediction.


The goal is to introduce beginners to deep learning concepts with an intuitive example.
---

## 📌 Key Features

- 📚 Loads the MNIST dataset (60,000 training + 10,000 test images)
- ⚙️ Preprocesses images by normalizing pixel values
- 🧠 Builds a simple neural network with:
  - 1 Flatten layer
  - 1 Dense hidden layer with ReLU activation
  - 1 Dense output layer with Softmax activation
- 🚀 Trains the model for 5 epochs
- 📊 Evaluates the model on the test set
- 🖼 Visualizes predictions with Matplotlib

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow 2.x
- Keras API
- Matplotlib
- NumPy

---

## 🧪 Dataset

**MNIST Handwritten Digit Dataset**  
Provided by: [Yann LeCun](http://yann.lecun.com/exdb/mnist/)  
- Images: 28×28 grayscale
- Labels: 0 to 9 (digit classes)

Loaded using:

```python
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

---

## 🧾 Installation & Usage

### ✅ Prerequisites

Make sure you have the following installed:

```bash
python3 --version
pip install tensorflow matplotlib numpy
```

### ▶️ Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yashpalsince2004/Handwritten_digit-Clasification_MINST.git
cd handwritten-digit-classification
```

2. Run the script:

```bash
python Day4_Handwritten_Digit_Classification_MNIST.py
```

---

## 🧠 Model Architecture

```text
Input: 28x28 image → Flatten → 784 neurons
↓
Dense Layer: 256 neurons, ReLU
↓
Output Layer: 10 neurons, Softmax
```

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

---

## 📈 Sample Output

- Prints training and test accuracy.
- Displays an image with predicted and actual labels.

```bash
Training data shape: (60000, 28, 28)
Epoch 1/5
...
Test Accuracy: 0.974
```

🖼 Then opens a window showing the first test image and prediction.

---

## 📌 Sample Visualization

```python
plt.imshow(X_test[0], cmap='gray')
plt.title(f"Predicted: {tf.argmax(predictions[0]).numpy()}, Actual: {y_test[0]}")
plt.show()
```

---

## 📤 Contributing

Contributions are welcome! If you’d like to improve this model or add features like CNN, dropout, or save/load models:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes
4. Push to the branch: `git push origin feature-name`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Yash Pal**  
🎓 Computer Science (AI/ML) Student  
🌐 [LinkedIn](https://www.linkedin.com/in/yash-pal-since2004) | 🧠 Passionate about AI & Deep Learning
