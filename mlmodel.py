import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('FuelConsumption.csv')

selected_features = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
data = data[selected_features + ['CO2EMISSIONS']]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = train_data[selected_features]
y_train = train_data['CO2EMISSIONS']

X_test = test_data[selected_features]
y_test = test_data['CO2EMISSIONS']

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)


poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)


print('Correlation =', poly_model.score(X_train_poly, y_train))
print('Correlation =', poly_model.score(X_test_poly, y_test))


with open('model.pickle', 'wb') as model_file:
    pickle.dump(poly_model, model_file)

print("Modèle sauvegardé avec succès dans le fichier 'model.pickle'")
