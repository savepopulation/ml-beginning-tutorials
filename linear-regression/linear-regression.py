import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Reading data
names = ["fiyat","metrekare"]
dataset = pd.read_csv("data.csv")

# Data
print(dataset)

# Check shape
print(dataset.shape)

# Check Description
print(dataset.describe())

# Show scatter matrix of data set
scatter_matrix(dataset)
plt.show()

# Converting to numpy arrays
x = dataset[names[0]]
y = dataset[names[1]]

plt.scatter(x,y)
plt.show()

x1 = np.array(x)
y1 = np.array(y)

# Calculate coefficients
m,b = np.polyfit(x1,y1,1)
a = np.arange(150)

plt.scatter(x,y)
plt.plot(m*a+b)

# Get intput and predict
z = int(input("metrekare?"))
prediction = m*z+b
print(prediction)

plt.scatter(z,prediction,c="red",marker=">")
plt.show()

print("y=",m,"x+",b)
