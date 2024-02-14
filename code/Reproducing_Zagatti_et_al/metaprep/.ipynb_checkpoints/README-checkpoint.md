# MetaPrep: Optimizing the Data Preparation via Meta-Learning
------------------

MetaPrep is a tool developed in Python for preprocessing and automatic data cleaning based on meta-learning, a technique that learns from past experiences. Thus, when presented a new dataset to be cleaned, the tool selects the ideal techniques for the types of data present and returns 5 pipeline recommendations to be used.

In addition to automatic preprocessing, MetaPrep is malleable, allowing the developer to define his own pipeline according to the techniques present in the algorithm.

![Execution-pipeline2](https://user-images.githubusercontent.com/50921477/106387927-dbf0f280-63ba-11eb-8ca1-7358b30e31f9.png)

#### Techniques present in MetaPrep:
* **Imputation of null data:**
  - Deletion case, mean, median and most frequent.
* **Standardization and normalization:**
  - Standard Scaler, minmax and normalizer;
* **Categorical-numerical transformation:**
  - One-Hot-Encoding and label encoder;
* **Class balancing:**
  - Oversampling and undersampling.

# Requirements
----------------------
```
Python >= 3.7
Pandas >= 1.2.1
Numpy >= 1.19.2
Pickle >= 4.0
Scikit-learn >= 0.24.1
Imbalanced-learn == 0.7.0
Pymfe == 0.4
```

# Utilization
------------------

Example with [Titanic Dataset](https://www.kaggle.com/c/titanic)

#### First, instantiate the cleaning method:

```Python
from metaprep import cleaner
data = cleaner()
```

#### Load the model:

```Python
data.load_csv('csv_results/train.csv')
```

Or for dataframes in memory:

```Python
import pandas as pd
df = pd.read_csv('csv_results/train.csv')
data.set_dataframe(df)
```

#### Applying automatic data preparation:

```Python
results = data.meta_prep(target='Survived', verbose=1) 
# This first method return a list with 5 best pipelines
data.preprocess(pipeline=results[0], target='Survived', verbose=1)
# This method realizes the data preparation with the selected pipeline
```

#### Test the dataset preprocessed:

```Python
data.test(algorithm='rf', target='Survived', verbose=1)
# It's possible utilize Random Forest as 'rf' or Support Vector Machine as 'svm'
```

#### Getting the preprocessed dataset:

```Python
df = data.get_dataframe()
```
