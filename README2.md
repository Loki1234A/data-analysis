# Data Analysis and Visualization

This Python script performs data analysis and visualization using various statistical techniques and plots. The script reads a dataset from a CSV file and provides functionality to explore and analyze the data through probability mass functions (PMF), probability density functions (PDF), cumulative distribution functions (CDF), scatter plots, and correlation matrices.

The following tasks are performed:
- Listing data fields and their types
- Checking for missing or infinite values
- Computing descriptive statistics (max, min, average, variance)
- Calculating and plotting PMF/PDF and CDF for each field
- Conditional PMF/PDF calculations based on class labels (anomaly or normal)
- Creating scatter plots to visualize relationships between fields
- Calculating joint PMF/PDF of two fields
- Analyzing conditional joint distributions
- Computing correlations between data fields
- Identifying fields dependent on the type of attack

## Prerequisites

Before running the script, ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- prettytable
- scipy

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn prettytable scipy
```

## Functions

### 1. `maketable(data)`
Creates a formatted table for displaying data using the PrettyTable library.

**Parameters:**
- `data`: A dictionary containing column names and their corresponding values.

### 2. `plot_pmf(df, column)`
Plots the Probability Mass Function (PMF) of a given categorical column.

**Parameters:**
- `df`: The DataFrame containing the data.
- `column`: The categorical column for which the PMF is plotted.

### 3. `plot_pdf(df, column)`
Plots the Probability Density Function (PDF) of a given numerical column using Kernel Density Estimation (KDE).

**Parameters:**
- `df`: The DataFrame containing the data.
- `column`: The numerical column for which the PDF is plotted.

### 4. `plot_cdf(df, column)`
Plots the Cumulative Distribution Function (CDF) of a given numerical column.

**Parameters:**
- `df`: The DataFrame containing the data.
- `column`: The numerical column for which the CDF is plotted.

### 5. `plotscatter(numericalcols)`
Generates a scatter plot between two randomly selected numerical columns.

**Parameters:**
- `numericalcols`: List of numerical column names in the DataFrame.

### 6. `analyze_and_plot_joint_distribution(df, numericalcols, categoricalcols)`
Analyzes and plots the joint distributions for selected pairs of categorical and numerical fields.

**Parameters:**
- `df`: The DataFrame containing the data.
- `numericalcols`: List of numerical column names.
- `categoricalcols`: List of categorical column names.

### 7. `analyze_and_plot_distributions(df, class_column)`
Analyzes and plots the distributions (PMF, PDF, CDF) for each column in the DataFrame.

**Parameters:**
- `df`: The DataFrame containing the data.
- `class_column`: The column name representing the class for conditional analysis.

### 8. `analyze_and_plot_conditional_joint_distribution(df, class_column, numericalcols, categoricalcols)`
Analyzes and plots the conditional joint distributions for pairs of fields given a class.

**Parameters:**
- `df`: The DataFrame containing the data.
- `class_column`: The column name representing the class for conditional analysis.
- `numericalcols`: List of numerical column names.
- `categoricalcols`: List of categorical column names.

## Data Inspection

The script also includes functionality to inspect the data, including:
- Listing all columns and their data types.
- Checking for missing and infinite values.
- Displaying the number of unique values per column.
- Summarizing basic statistics (max, min, mean, variance).
- Calculating the correlation matrix among numerical columns.

## Usage

1. Place your dataset (`Train_data.csv`) in the same directory as the script.
2. Run the script in your Python environment. It will read the dataset and perform analysis and visualization automatically.

## Notes

- Ensure the dataset is properly formatted and accessible.
- Customize the class column name based on your dataset for conditional analysis.

## Author

- Youssef Guirguis Bekheet Kirolous :)

