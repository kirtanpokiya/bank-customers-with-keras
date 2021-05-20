import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, -1]

geography = pd.get_dummies(X["Geography"], drop_first=True)
gender = pd.get_dummies(X["Gender"], drop_first=True)

X = pd.concat([X,geography],axis=1)
X = pd.concat([X,gender], axis=1)

X = X.drop(["Geography", "Gender"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units=6, kernel_initializer="he_uniform", activation="relu", input_dim=11))
model.add(Dense(units=6, kernel_initializer="he_uniform", activation="relu"))
model.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model_history = model.fit(X_train, y_train, validation_split=0.3, batch_size=10, nb_epoch=100)

print(model_history.history.keys())

plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.show()

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)

print("accuracy score  :" + str(score))