from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the home prices data
df = pd.read_csv('/content/drive/MyDrive/homeprices.csv')
print(df)

# Visualize the data
%matplotlib inline #imp in colab
plt.scatter(df.area, df.prices, color='red')

# Create a linear regression model (training) 
reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.prices)

# Make a prediction for a new area (3300) using inbuilt predict method
prediction = reg.predict([[3300]])
plt.plot(df.area, reg.predict(df[['area']]), color='blue')

# How the prediction is made by the model(manual approach)
m = reg.coef_
c = reg.intercept_
y = (m * 3300) + c

#printing outputs
print("Prediction:")
print(prediction)
print("Manually calculated prediction:")
print(y)

# Read the new areas from the 'area.csv' file
d = pd.read_csv('/content/drive/MyDrive/area.csv')
print(d)

# Use the model to predict prices for the new areas
p = reg.predict(d[['area']])
