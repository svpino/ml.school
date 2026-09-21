# Assignments

Complete the following assignments to reinforce the concepts we covered in this session. You don't have to work on every assignment. Pick the ones that you think will help you the most.

1. Create the `.env` file by running `just env` and verify the variables are loaded by running `echo $KERAS_BACKEND` and `echo $MLFLOW_TRACKING_URI`. 

1. Run `mlflow server --help` and identify the parameters for specifying the host and port. Run the MLflow server on a different port (e.g., 5500) bound to 127.0.0.1 and verify the UI at [`http://127.0.0.1:5500`](http://127.0.0.1:5500).

1. Extend the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook by calculating the percentage of missing values for each column, analyzing patterns in missing data, and investigating correlations between missing values.

1. For each numerical feature in the dataset, create histograms in the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook with different bin sizes, identify potential outliers using statistical methods, and compare distributions between different species.

1. Extend the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook with a few box plots for each numerical feature in the dataset grouped by species to compare the distributions and identify which features best distinguish each species.

1. Extend the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook by calculating summary statistics (mean, median, standard deviation) for each numerical feature broken down by species to understand species-specific characteristics.

1. Investigate the relationship between flipper length and body mass in the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook by creating a scatter plot and calculating the correlation coefficient for each species separately.

1. Extend the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook by creating frequency tables for all combinations of categorical variables (species, island, sex) to understand the data distribution across different groups.

1. Extend the [Exploratory Data Analysis](notebooks/eda.ipynb) notebook by creating histograms with overlapping distributions for each numerical feature, separated by species, to visually compare how each species differs across measurements.

1. Write a short Python script that logs an experiment to a local MLflow server. Create an experiment, start a run, log at least two parameters and one small artifact. Open the MLflow UI and verify that everything was logged correctly.

1. Start the MLflow server with a SQLite backend store and a filesystem artifact root. Run a Python script that logs an experiment, and confirm the database and artifacts were written to disk.