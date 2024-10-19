import matplotlib as plt
import numpy as np
import pandas as pd

#read data
df =  pd.read_csv('Train_data.csv')
#listing columns
columns = df.columns
print(columns, "\n")
#listing heading types
data_heads = df.dtypes
print(data_heads, "\n")

#check if there is missing/inf data
missing_data = df.isnull().sum()
print("Missing data per column:\n", missing_data, "\n")
# Select only numeric columns using `select_dtypes`
numeric_df = df.select_dtypes(include=[np.number])

# Check for infinite values column-wise and sum them
inf_data = np.isinf(numeric_df).sum()
print("Infinite data per column:\n", inf_data, "\n")
#then replacing the inf data for calculations
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#printing number of unique values
unique_values = df.nunique()
print("Number of unique values per column:\n", unique_values, "\n")
# Get a quick summary of basic statistics
summary_statistics = df.describe()
print("Summary statistics:\n", summary_statistics)

# Calculate each measure individually if needed
max_values = numeric_df.max()
min_values = numeric_df.min()
mean_values = numeric_df.mean()
variance_values = numeric_df.var()

print("Maximum values:\n", max_values)
print("Minimum values:\n", min_values)
print("Mean values:\n", mean_values)
print("Variance values:\n", variance_values)