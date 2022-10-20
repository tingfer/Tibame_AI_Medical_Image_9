import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a empty DataFrame
df = pd.DataFrame()
print(df)

# add a column
df['name'] = ['A', 'B', 'C']
print(df)

# add 2nd and 3rd column
df['height'] = [150, 180, 170]
df['weight'] = [45, 70, 64]
print(df)

# Create DataFrame with Values
df = pd.DataFrame(
    {
        'name': ['A', 'B', 'C'],
        'height': [150, 180, 170],
        'weight': [45, 70, 64]
    }
)
print(df)

# First 2 rows
print(df.head(2))

# Last 2 rows
print(df.tail(2))

# index and columns
print(df.index)
print(df.columns)

# convert to numpy array
df_numpy = df.to_numpy()
df_numpy
print(df_numpy.shape)
print(df_numpy)

print(df.describe())

# Columns
print(df['name'])
print(df['weight'])

# Rows
print(df[0:2])

# select by position
print(df.iloc[2])

# select first 2 rows and last column
print(df.iloc[:2, 2])

# select height > 160
print(df[df['height'] > 160])

df['height>160'] = df['height'] > 160
print(df)

df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
print(df)

df = pd.DataFrame(
    {
        'height': np.random.normal(loc=165, scale=10, size=100).astype(np.int32),
        'weight': np.random.normal(loc=60, scale=20, size=100).astype(np.int32),
    }
)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
print(df)

plt.plot(df['height'])
plt.show()

plt.hist(df['height'])
plt.show()

plt.hist(df['BMI'])
plt.show()

plt.scatter(df['height'], df['weight'])
plt.show()

df.to_csv('pandas.csv', index=None)

df_read = pd.read_csv('pandas.csv')
print(df_read)
