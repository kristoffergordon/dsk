# DSK: Data Science Toolkit

DSK is a comprehensive toolkit designed to streamline various data science tasks, from data preprocessing to visualization. It aims to provide data scientists and analysts with a set of tools to efficiently handle data, perform exploratory analysis, and generate insightful visualizations.

## Features

- **Data Preprocessing**: Functions to clean and prepare data for analysis, including handling missing values and reducing memory usage.
- **Visualization**: A suite of visualization tools built on top of matplotlib and seaborn for creating informative plots easily.

## Installation

DSK is available on PyPI. You can install it using pip:

```sh
pip install dsk
```

## Usage
Import the DSK package and use its modules as follows:

```python
import pandas as pd
from dsk import data, plots

# Example: Load and preprocess data
df = pd.read_csv('your_dataset.csv')
df_overview = data.data_overveiw(df)

# Example: Visualize data distribution
plots.plot_distribution(df, 'your_feature_column')
```
