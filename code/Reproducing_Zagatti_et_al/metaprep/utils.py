#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Develop by Fernando Rezende Zagatti

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from metaprep.imblearn.over_sampling import RandomOverSampler
from metaprep.imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from pymfe.mfe import MFE
import numpy as np
import pandas as pd
from pandas import read_csv, get_dummies

import numpy as np

def data_caracterization(data, target, tram='gray'):
    """This method performs the data characterization.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be characterized.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
    
    tram: :obj: 'str', optional
          If 'gray', then all categoric-type data will be binarized with a
          model matrix strategy. If 'one-hot', then all categoric-type
          data will be transformed using the k-1 one-hot encoding strategy
          (for a traditional one-hot encoding, the first column is dropped
          out). If 'one-hot-full', the strategy used is the one-hot encoding
          with all encoded features ('k' features for an attribute with 'k'
          unique values; not recommended due to multicollinearity problems
          due to the 'dummy variable trap'). If None, then the categorical
          attributes are not transformed.
          
    Returns
    -------
    
    ft: Pandas DataFrame,
        Returns a DataFrame with all the characteristics of the data, being 
        simple, statistical and information theory characteristics.
        
    References
    ----------
    
    .. [1] Edesio Alcobaça, Felipe Siqueira, Adriano Rivolli, Luís P. F. Garcia, 
       Jefferson T. Oliva, & André C. P. L. F. de Carvalho (2020). MFE: Towards 
       reproducible meta-feature extraction. Journal of Machine Learning Research, 
       21(111), 1-5.
    """
    sum_null = data.isna().sum().sum()
    rows_null = data.shape[0] - data.dropna().shape[0]
    samples, dimensionality = data.shape
    dimensionality = dimensionality - 1
    data = data.dropna(inplace=False)
    y_target = data[target]
    data = data.drop([target], axis=1)
    y_target = np.array(y_target)
    data = np.array(data)
    mfe = MFE(groups=["general", "statistical", "info-theory"])
    mfe.fit(X = data, y = y_target, transform_cat = tram)
    ft = mfe.extract()
    names = ft[0]
    names.insert(0, 'dimensionality')
    names.insert(0, 'rows_null')
    names.insert(0, 'total_null')
    values = ft[1]
    values.insert(0, dimensionality)
    values.insert(0, rows_null)
    values.insert(0, sum_null)
    ft = pd.DataFrame([values], columns=names)
    ft = ft.replace(np.nan, -999)
    ft = ft.replace(np.inf, -9999)
    ft = ft.drop(columns=['eigenvalues.mean'])
    return ft    


def separate(data, target=None): 
    """This method performs the separation of numerical and categorical data.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str', optional
            target of the dataset that has been passed.
          
    Returns
    -------
    
    num: Pandas DataFrame,
         DataFrame with all numeric data.
    
    cat: Pandas DataFrame,
         DataFrame with all categorical data. 
    
    y_target: Pandas DataFrame,
              DataFrame with data target
    
    num_columns: :obj: 'list' of str,
                 List with the name of all numerical data.
    
    cat_columns: :obj: 'list' of str,
                 List with the name of all categorical data.
    """
    data = data.reset_index(drop=True)
    all_columns = list(data.columns)
    num_columns = data.select_dtypes(include=np.number).columns.tolist() 
    cat_columns = [x for x in all_columns if not x in num_columns]   
    if target is not None:
        y_target = data[target] 

        if target in num_columns: 
            num_columns.remove(target)
        if target in cat_columns: 
            cat_columns.remove(target)
    else:
        y_target = None
    
    num, cat = None, None
    for column in num_columns: 
        if(num is None):
            num = data[column]
        else:
            num = pd.concat([num, data[column]], axis=1)
    for column in cat_columns: 
        if(cat is None):
            cat = data[column]
        else:
            cat = pd.concat([cat, data[column]], axis=1)
            
    if num is not None:
        num = num.reset_index(drop=True)
    if cat is not None:
        cat = cat.reset_index(drop=True)
    return num, cat, y_target, num_columns, cat_columns


def one_hot_encoder(data, target):
    """This method performs the One-Hot-Encoding.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all categorical data transformed into numeric columns.
    """
    x = data.drop([target], axis=1)
    y = data[target]
    data = get_dummies(x)
    data = pd.concat([data, y], axis=1)
    return data


def label_encoder(data, target):
    """This method performs the Label Encoder.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all categorical data transformed into numerical data.
    """
    num, cat, y_target, num_columns, cat_columns = separate(data, target)
    if type(cat) is pd.Series:
        cat = pd.DataFrame(cat, columns = cat_columns)
    if cat is not None:
        data = cat.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)), axis=0, result_type='expand')
        if num is not None:
            data = pd.concat([num, data], axis=1)
    else: 
        data = num
    data = pd.concat([data, y_target], axis=1)
    return data


def minmax(data, target):
    """This method performs the MinMax normalization.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all numerical data normalized with the MinMax technique.
    """
    num, cat, y_target, num_columns, cat_columns = separate(data, target)
    scaler = MinMaxScaler()
    if num is not None:
        data = scaler.fit_transform(num) 
        data = pd.DataFrame(data, columns=num_columns)
        if cat is not None:
            data = pd.concat([data, cat], axis=1) 
    else:
        data = cat
    data = pd.concat([data, y_target], axis=1)
    return data


def standard_scaler(data, target):
    """This method performs the Standard Scaler standardization.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all numerical data standardized with the Standard 
          Scaler technique.
    """
    num, cat, y_target, num_columns, cat_columns = separate(data, target)
    scaler = StandardScaler()
    if num is not None:
        data = scaler.fit_transform(num) 
        data = pd.DataFrame(data, columns=num_columns)
        if cat is not None:
            data = pd.concat([data, cat], axis=1)
    else:
        data = cat
    data = pd.concat([data, y_target], axis=1)
    return data


def normalizer(data, target):
    """This method performs the Normalizer normalization.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all numerical data normalized with the Normalizer 
          technique.
    """
    try:
        num, cat, y_target, num_columns, cat_columns = separate(data, target)
        if num is not None:
            num = num.reset_index(drop=True)
            transformer = Normalizer().fit(num)
            data = transformer.transform(num)
            data = pd.DataFrame(data, columns=num_columns)
            if cat is not None:
                cat = cat.reset_index(drop=True)
                data = pd.concat([data, cat], axis=1)
        data = pd.concat([data, y_target], axis=1)
        return data
    except OSError as err:
        print("OS error: {0}".format(err))
        return data
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        return data
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return data


def imputation_mean(data, target):
    """This method performs the imputation mean.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all missing data filled with the mean.
    """
    if data.isna().sum().sum() > 0:
        x = data.drop([target], axis=1)
        y = data[target]
        data = x.fillna(x.mean())
        data = pd.concat([data, y], axis=1)
    return data


def imputation_median(data, target):
    """This method performs the imputation median.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all missing data filled with the median.
    """
    if data.isna().sum().sum() > 0:
        x = data.drop([target], axis=1)
        y = data[target]
        data = x.fillna(x.median())
        data = pd.concat([data, y], axis=1)
    return data


def imputation_deletion_case(data, target): 
    """This method performs the imputation deletion case.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all samples with missing data excluded 
          by deletion case.
    """
    if data.isna().sum().sum() > 0:
        data.dropna(inplace=True)
        data = data.reset_index(drop=True)
    return data


def imputation_most_frequent(data, target):
    """This method performs the imputation most frequent.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          DataFrame with all missing data filled with the most frequent.
    """
    if data.isna().sum().sum() > 0:
        columns = data.columns
        imp = SimpleImputer(strategy='most_frequent')
        imp.fit(data)
        data = imp.transform(data)
        data = pd.DataFrame(data, columns = columns)
    return data

#Class balancing
def undersampling(data, target):
    """This method performs the Undersampling.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          Returns a DataFrame with all data balanced through undersampling.
    """    
    under = RandomUnderSampler(random_state=42)
    y = data[target]
    X = data.drop([target], axis=1)
    all_columns = list(X.columns)
    try:
        y = y.to_numpy()
        X = X.to_numpy()
        if isinstance(y, object):
            y=y.astype('str')
        X, y = under.fit_resample(X, y)
        y = pd.DataFrame(data=y, columns=[target])
        X = pd.DataFrame(data=X, columns=all_columns)
        data = pd.concat([X, y], axis=1)
        return data
    except:
        return data

def oversampling(data, target):
    """This method performs the Oversampling.
    
    Parameters
    ----------
    
    data: Pandas DataFrame, 
          data that will be separated.
    
    target: :obj: 'str',
            target of the dataset that has been passed.
          
    Returns
    -------
    
    data: Pandas DataFrame,
          Returns a DataFrame with all data balanced through oversampling.
    """
    over = RandomOverSampler(random_state=42)
    y = data[target]
    X = data.drop([target], axis=1)
    all_columns = list(X.columns)
    try:
        if isinstance(y, object):
            y=y.astype('str')
        X, y = over.fit_resample(X, y)
        y = pd.DataFrame(data=y, columns=[target])
        X = pd.DataFrame(data=X, columns=all_columns)
        data = pd.concat([X, y], axis=1)
        return data
    except:
        return data