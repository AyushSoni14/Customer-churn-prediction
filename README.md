# Customer Churn Prediction using ANN 

## Overview
This project demonstrates how to predict customer churn using an Artificial Neural Network (ANN) with **TensorFlow 2.0, Keras, and Python**. The model is trained on customer data to classify whether a customer will churn or not, based on various features such as tenure, monthly charges, and service usage patterns.

## Features
- Data preprocessing and feature encoding
- Exploratory Data Analysis (EDA) with visualization
- Building and training an ANN model using TensorFlow/Keras
- Evaluating the model with accuracy metrics and confusion matrix

## Dataset
The dataset consists of customer information with the following key features:
- **Tenure**: Duration of customer relationship
- **MonthlyCharges**: Customer's monthly bill
- **Services Used**: Information about services subscribed by the customer
- **Churn**: Whether the customer left (Yes/No)

## Installation
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install tensorflow keras pandas matplotlib seaborn
```

## Data Preprocessing
```python
# Converting categorical columns to numerical
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)
```

## Exploratory Data Analysis (EDA)
### Tenure vs. Churn Visualization
```python
plt.xlabel("Tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
```

### Monthly Charges vs. Churn Visualization
```python
plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
```

## Building the ANN Model
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)
```

## Model Evaluation
```python
import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
```

## Conclusion
This project successfully demonstrates how an ANN can be used to predict customer churn based on service usage patterns and customer demographics. By leveraging **TensorFlow and Keras**, we can build an effective deep learning model for business insights and customer retention strategies.



