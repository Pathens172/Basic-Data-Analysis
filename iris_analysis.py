# We need to import the magic book pandas
import pandas as pd

# Now we load the Iris dataset (you can think of it as opening your notebook)
from sklearn.datasets import load_iris
data = load_iris()

# Now we make the dataset look like a table with rows and columns
df = pd.DataFrame(data=data['data'], columns=data['feature_names'])

# We also add a column called "species" to say what type of flower it is
df['species'] = data['target']
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Look at the first few rows of the table to see what we have
print(df.head())

# Check what the data looks like and what kind of things we have in each column
print(df.info())

# Check if there are any missing pieces of information
print(df.isnull().sum())

# If there were missing values, we could drop them:
df = df.dropna()

# Or, we could fill missing values with something (like the average number):
# df = df.fillna(df.mean())

# This gives us the average, min, max, and other stats for each number in the table
print(df.describe())

# Group the flowers by species and find the average size for each group
grouped_data = df.groupby('species').mean()

# Print the results to see the comparison
print(grouped_data)

import matplotlib.pyplot as plt

# Line chart showing petal length vs petal width
plt.plot(df['sepal length (cm)'], df['sepal width (cm)'])
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Bar chart showing the average petal length for each species
grouped_data['petal length (cm)'].plot(kind='bar')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram showing the distribution of petal lengths
df['petal length (cm)'].plot(kind='hist', bins=20)
plt.title('Distribution of Petal Lengths')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot showing the relationship between sepal length and petal length
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

