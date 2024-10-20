import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from scipy.stats import gaussian_kde

#read data
df = pd.read_csv('Train_data.csv')

#beautifying the outputs
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Increase the display width
pd.set_option('display.float_format', '{:.2f}'.format)  # Display floats with 2 decimal places
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

plt.close('all')


def maketable(data):
    table = PrettyTable()
    for col, val in data.items():
        table.add_row([col, val])
    return table

#plotting details
def plot_pmf(df, column):
    pmf = df[column].value_counts(normalize=True)  # Normalize to get probabilities
    plt.figure(figsize=(15, 4))
    pmf.plot(kind='bar')
    plt.title(f'PMF of {column}')
    plt.xlabel(column)
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pdf(df, column):
    plt.figure(figsize=(8, 4))
    sns.kdeplot(df[column], bw_adjust=0.5, warn_singular=False)  #kernel density estimation
    plt.title(f'PDF of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()


def plot_cdf(df, column):
    # Drop NaN values + !Sorting!
    data = df[column].dropna().sort_values()

    # Array for the cumulative probabilities
    cdf = np.arange(1, len(data) + 1) / len(data)

    # Plot the CDF
    plt.figure(figsize=(8, 4))
    plt.plot(data, cdf, linestyle='-', marker='')  # Use a line to connect points
    plt.title(f'Cumulative Distribution Function (CDF) of {column}')
    plt.xlabel(column)
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotscatter(numericalcols):
    for i in range(0, len(numericalcols) - 1, 2):  # Iterating by steps of 2, not going over bounds
        x_field = numericalcols[i] # First column
        y_field = numericalcols[i + 1] # Second column

        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_field], df[y_field], alpha=0.5, color='blue')  # Use alpha for point transparency
        plt.title(f'Scatter Plot between {x_field} and {y_field}')
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.grid(True)
        plt.tight_layout()
        plt.show()  # Display the plot


# def analyze_and_plot_joint_distribution(df, numericalcols, categoricalcols):
#
#     # Function to calculate and plot joint PMF for categorical fields
#     def plot_joint_pmf(x_field, y_field):
#         joint_counts = pd.crosstab(df[x_field], df[y_field])
#         joint_pmf = joint_counts / joint_counts.sum().sum()  # Normalize to get PMF
#
#         plt.figure(figsize=(10, 6))
#         sns.heatmap(joint_pmf, annot=True, cmap='Blues')
#         plt.title(f'Joint PMF of {x_field} and {y_field}')
#         plt.xlabel(y_field)
#         plt.ylabel(x_field)
#         plt.show()
#
#     # Function to calculate and plot joint PDF for numerical fields
#     def plot_joint_pdf(x_field, y_field):
#         if df[x_field].nunique() < 2 or df[y_field].nunique() < 2:
#             print(f"Skipping {x_field} and {y_field} due to insufficient unique values.")
#             return  # Skip plotting if not enough unique values
#
#         plt.figure(figsize=(10, 6))
#
#         # Filter duplicates to avoid contour issues
#         filtered_df = df[[x_field, y_field]].drop_duplicates()
#
#         # Use KDE for plotting
#         sns.kdeplot(x=filtered_df[x_field], y=filtered_df[y_field], fill=True, cmap='Blues', thresh=0, bw_adjust=0.5)
#         plt.title(f'Joint PDF of {x_field} and {y_field}')
#         plt.xlabel(x_field)
#         plt.ylabel(y_field)
#         plt.grid(True)
#         plt.show()
#
#     # Loop through numeric columns to calculate joint PDFs
#     for i in range(0, len(numeric_cols) - 1, 2):
#         x_field = numericalcols[i]
#         y_field = numericalcols[i + 1]
#         plot_joint_pdf(x_field, y_field)
#
#     # Loop through categorical columns to calculate joint PMFs
#     for i in range(0, len(categorical_cols) - 1, 2):
#         x_field = categoricalcols[i]
#         y_field = categoricalcols[i + 1]
#         plot_joint_pmf(x_field, y_field)

def analyze_and_plot_joint_distributions(df,numericalcols,categoricalcols):
    """
    Analyzes the dataframe and plots joint distributions for pairs of fields.
    Handles numerical-numerical, categorical-categorical, and numerical-categorical cases.
    """

    # Iterate over pairs of numeric columns (0-1, 2-3, etc.)
    for i in range(0, max(len(numericalcols), len(categoricalcols)) - 1, 2):
        if i < len(numericalcols) - 1:
            x_field = numericalcols[i]
            y_field = numericalcols[i + 1]

            # Drop NaN values for the selected fields
            data = df[[x_field, y_field]].dropna()

            # Get x and y values
            x = data[x_field]
            y = data[y_field]

            # Create a figure for 3D plotting
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Calculate and plot the joint distribution as a 3D surface
            sns.histplot(data, x=x_field, y=y_field, bins=30, pthresh=0.1, cmap='viridis', cbar=True, ax=ax)
            ax.set_xlabel(x_field)
            ax.set_ylabel(y_field)
            ax.set_zlabel('Frequency')
            plt.title(f'3D Joint Distribution of {x_field} and {y_field}')

            # Show the plot
            plt.show()

        if i < len(categoricalcols) - 1:
            cat1 = categoricalcols[i]
            cat2 = categoricalcols[i + 1]

            # Drop NaN values for the selected fields
            data = df[[cat1, cat2]].dropna()

            # Calculate and plot joint PMF
            joint_pmf = data.groupby([cat1, cat2]).size() / len(data)
            joint_pmf = joint_pmf.unstack(fill_value=0)

            # Create a heatmap for the joint PMF
            plt.figure(figsize=(10, 6))
            sns.heatmap(joint_pmf, annot=True, fmt=".2f", cmap='Blues')
            plt.title(f'Joint PMF of {cat1} and {cat2}')
            plt.xlabel(cat2)
            plt.ylabel(cat1)
            plt.show()

        if i < len(numeric_cols) and i < len(categorical_cols):
            num_field = numeric_cols[i]
            cat_field = categorical_cols[i]

            # Drop NaN values for the selected fields
            data = df[[num_field, cat_field]].dropna()

            # Create a bar plot for numerical values across categories
            plt.figure(figsize=(10, 6))
            sns.histplot(data, x=num_field, hue=cat_field, multiple="stack", kde=True)
            plt.title(f'Distribution of {num_field} by {cat_field}')
            plt.xlabel(num_field)
            plt.ylabel('Frequency')
            plt.show()


# Checking which columns are pdf and which are pmf+cdf and plotting
def analyze_and_plot_distributions(df, class_column):
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            # Categorical data: calculate and plot PMF
            plot_pmf(df, column)
        elif np.issubdtype(df[column].dtype, np.number):
            # Continuous data: calculate and plot PDF and CDF
            # Drop NaN values for KDE
            plot_pdf(df.dropna(subset=[column]), column)  # Plot PDF

            # Now add logic to calculate and plot conditional PDFs
            normal_data = df[df[class_column] == 'normal'][column].dropna()
            anomaly_data = df[df[class_column] == 'anomaly'][column].dropna()

            # Create a new plot for conditional PDFs
            plt.figure(figsize=(10, 6))

            # Plot PDF for the entire dataset
            sns.kdeplot(df[column].dropna(), color='blue', label='PDF of All Data', linewidth=2, warn_singular=False)

            # Plot PDF for normal class
            sns.kdeplot(normal_data, color='green', label='PDF of Normal Class', linewidth=2, warn_singular=False)

            # Plot PDF for anomaly class
            sns.kdeplot(anomaly_data, color='red', label='PDF of Anomaly Class', linewidth=2, warn_singular=False)

            # Set titles and labels
            plt.title(f'Conditional PDF of {column} Given Class')
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.legend()  # This will now have the correct entries
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot the CDF
            plot_cdf(df, column)  # Plot the CDF for the whole dataset
        else:
            print(f"\nSkipping column {column} with type {df[column].dtype}")


analyze_and_plot_distributions(df, 'class')
analyze_and_plot_joint_distributions(df, numeric_cols, categorical_cols)
plotscatter(numeric_cols)


#listing columns
columns = df.columns
print(columns, "\n")
#listing heading types
data_heads = df.dtypes
print(data_heads, "\n")

#check if there is missing/inf data + print in a table
missing_data = df.isnull().sum()
table = maketable(missing_data)
table.field_names = ["Column", "Missing values"]
print("Number of missing values per column:\n", table, "\n")


# Select only numeric columns using `select_dtypes`
numeric_df = df.select_dtypes(include=[np.number])

# Check for infinite values column-wise and sum them
inf_data = np.isinf(numeric_df).sum()
table = maketable(inf_data)
table.field_names = ["Column", "Infinite values"]
print("Number of infinite values per column:\n", table, "\n")

#then replacing the inf data for calculations
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#printing number of unique values
unique_values = df.nunique()

# Create a table for unique values
table = maketable(unique_values)
table.field_names = ["Column", "Unique values"]
print("Number of unique values per column:\n", table, "\n")


# Repeat for other metrics like summary statistics or missing data

# Get a quick summary of basic statistics
summary_statistics = df.describe()

# Create a table for summary statistics values
table = maketable(summary_statistics)
table.field_names = ["Column", "Basic Statistics"]
print("Summary of data:\n", table, "\n")


# Calculate each measure individually if needed
max_values = numeric_df.max()
min_values = numeric_df.min()
mean_values = numeric_df.mean()
variance_values = numeric_df.var()

table1 = maketable(max_values)
table2 = maketable(min_values)
table3 = maketable(mean_values)
table4 = maketable(variance_values)

table1.field_names = ["Column", "Max values"]
table2.field_names = ["Column", "Min values"]
table3.field_names = ["Column", "Mean values"]
table4.field_names = ["Column", "Variance values"]

print("Maximum values:\n", table1, "\n")
print("Minimum values:\n", table2, "\n")
print("Mean values:\n", table3, "\n")
print("Variance values:\n", table4, "\n")
