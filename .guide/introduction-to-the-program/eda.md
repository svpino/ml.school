# Exploratory Data Analysis

Exploratory Data Analysis (EDA) is an essential step for understanding the structure and characteristics of the data we'll use to build a model.

During this process, we will identify any anomalies with the data and try to discover patterns and insights that will guide the project. Exploratory Data Analysis also helps validating assumptions, informing feature engineering, and ensuring a solid foundation for modeling.

The Exploratory Data Analysis notebook loads the initial dataset and systematically explores its structure and content. It looks at the distribution of the target variable, the distribution of the features, and the relationships between them.

You can execute the notebook from the command line by running the following command:

```shell
uv run -- jupyter execute notebooks/eda.ipynb
```

You can also open the notebook in Jupyter and run its cells interactively. 

Some of the decisions that we'll make when preprocessing the data will be based on the insights gained during the Exploratory Data Analysis process.