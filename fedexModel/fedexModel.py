import pandas as pd
from tensorflow import keras 
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split


df = pd.read_csv('fedex_data.csv')

dummies = pd.get_dummies(df, columns=['Day of the Week', 'Weather Condition', 'Delivery Area'])

# x = features used to predict, y = predicted data
x = dummies.drop(['Number of Packages'], axis =1).values
y = dummies['Number of Packages'].values

#define train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100, batch_size=16)
mse = model.evaluate(x_test, y_test)

X_new = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 55]])  # Example input for Driver ID 1, Monday, Sunny, Urban
predicted_packages = model.predict(X_new)
print('Predicted number of packages:', predicted_packages[0][0])