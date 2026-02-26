Hyperspectral Metal Toxicity Classification Using Artificial Neural Networks

Machine learning pipeline for detecting heavy metal toxicity (Cadmium) in kale and basil using VNIR hyperspectral reflectance data.

Originally developed at Purdue University – Institute for Plant Sciences
Edited and extended by Maria Paula Zea

🔬 Project Overview

This project uses hyperspectral reflectance data (CEPF platform) to:

Estimate heavy metal concentration (Cd)

Classify toxicity levels

Compare spectral preprocessing strategies

Train and evaluate an Artificial Neural Network (ANN) classifier

The model transforms reflectance into absorbance, applies spectral preprocessing techniques, and trains a neural network to detect toxic thresholds.

📊 Data Processing Pipeline
1️⃣ Spectral Preprocessing

Several preprocessing strategies were evaluated:

Reflectance → Absorbance conversion

Standard Normal Variate (SNV)

Savitzky-Golay 1st derivative

Detrending

Combined normalization + derivative

Example:

X_corr = np.log10(1/X)  # Reflectance to absorbance

# SNV normalization
uX = np.mean(X_corr,1)
deltaX = np.std(X_corr,1)
X1 = (np.array(X_corr)-uX.values.reshape(-1,1))/deltaX.values.reshape(-1,1)

# First derivative
X2 = sp.savgol_filter(X_corr, window_length=5, polyorder=3, deriv=1, axis=0)
2️⃣ Toxicity Threshold Definition

Binary classification was performed using a toxicity cutoff:

indy = y >= 1
y = indy
3️⃣ ANN Model Development

Model implemented using Scikit-Learn:

from sklearn.neural_network import MLPClassifier

regr = MLPClassifier(
    solver='lbfgs',
    alpha=1e-10,
    hidden_layer_sizes=(50,),
    random_state=1,
    max_iter=1000
)

regr.fit(X_train, y_train)
4️⃣ Model Evaluation

Train/Test split (75/25 stratified)

Confusion Matrix

Accuracy score

cm = confusion_matrix(y_test, pred_test)
ConfusionMatrixDisplay(cm).plot()

print('Score testing: ', regr.score(X_test, y_test))
📈 Results

Neural network successfully classified toxic vs non-toxic samples.

Spectral preprocessing significantly improved model stability.

First derivative + SNV normalization showed improved separation.

🛠 Technologies Used

Python

NumPy

Pandas

Scikit-Learn

SciPy

Matplotlib
