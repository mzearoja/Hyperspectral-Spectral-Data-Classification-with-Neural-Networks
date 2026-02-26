# Hyperspectral Metal Toxicity Classification Using ANN

Artificial Neural Network model to classify cadmium toxicity levels in kale and basil using VNIR hyperspectral data.

Developed at Purdue University – Institute for Plant Sciences  
Edited and extended by Maria Paula Zea


## Project Overview

This project uses hyperspectral reflectance data to:

- Estimate heavy metal contamination (Cd)
- Apply spectral preprocessing techniques
- Train a neural network classifier
- Evaluate model performance



## Spectral Preprocessing

Reflectance → Absorbance transformation:

```python
X_corr = np.log10(1/X)
```

Standard Normal Variate (SNV):

```python
uX = np.mean(X_corr,1)
deltaX = np.std(X_corr,1)
X1 = (np.array(X_corr)-uX.values.reshape(-1,1))/deltaX.values.reshape(-1,1)
```

Savitzky-Golay first derivative:

```python
X2 = sp.savgol_filter(X_corr, window_length=5, polyorder=3, deriv=1, axis=0)
```

---

## Neural Network Model

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    solver='lbfgs',
    hidden_layer_sizes=(50,),
    max_iter=1000,
    random_state=1
)

model.fit(X_train, y_train)
```

---

## Model Evaluation

```python
from sklearn.metrics import confusion_matrix

pred_test = model.predict(X_test)
cm = confusion_matrix(y_test, pred_test)
print("Test Accuracy:", model.score(X_test, y_test))

--

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

---

##  Applications

- Precision agriculture
- Environmental contamination monitoring
- Spectral AI modeling

