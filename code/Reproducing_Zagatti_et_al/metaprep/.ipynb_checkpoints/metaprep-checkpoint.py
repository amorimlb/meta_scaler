#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Develop by Fernando Rezende Zagatti

import pandas as pd
import pickle

from metaprep.utils import one_hot_encoder, label_encoder
from metaprep.utils import minmax, standard_scaler, normalizer
from metaprep.utils import imputation_mean, imputation_median, imputation_deletion_case, imputation_most_frequent
from metaprep.utils import oversampling, undersampling
from metaprep.utils import separate
from metaprep.utils import data_caracterization
# from utils import one_hot_encoder, label_encoder
# from utils import minmax, standard_scaler, normalizer
# from utils import imputation_mean, imputation_median, imputation_deletion_case, imputation_most_frequent
# from utils import oversampling, undersampling
# from utils import separate
# from utils import data_caracterization
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class cleaner(object):
    
    def __init__(self):
        dataframe = pd.DataFrame()
        
    @classmethod
    def load_csv(cls, file_path, missing_headers=False, sep=','):
        if missing_headers:
            data = pd.read_csv(file_path, header=None, sep=sep)
        else:
            data = pd.read_csv(file_path, header=0, sep=sep)
        cls.dataframe = data

    @classmethod
    def describe(cls):
        print(cls.dataframe.describe())

    PIPELINE_OPTIONS = {
        "imputation_mean": imputation_mean,
        "imputation_median": imputation_median,
        "imputation_deletion_case": imputation_deletion_case,
        "imputation_most_frequent": imputation_most_frequent,
        "one_hot_encoder": one_hot_encoder,
        "label_encoder": label_encoder,
        "minmax": minmax,
        "normalizer": normalizer,
        "standard_scaler": standard_scaler,
        "oversampling": oversampling,
        "undersampling": undersampling
    }
    
    @classmethod
    def print_dataframe(cls):
        print(cls.dataframe)
    
    @classmethod
    def get_dataframe(cls):
        return cls.dataframe
    
    @classmethod 
    def set_dataframe(cls, data):
        cls.dataframe = data

    @classmethod
    def preprocess(cls, pipeline, target, verbose=0):
        print('=============================== RUNNING THE DATA PREPARATION ===============================\n')
        print('Defined pipeline:', pipeline, '\n')
        if(verbose > 0):
            print('Starting data preparation...\n')
        for stage in pipeline:
            if stage in cls.PIPELINE_OPTIONS:
                if(verbose > 0):
                    print("Stage --> ", stage)
                cls.dataframe = cls.PIPELINE_OPTIONS[stage](cls.dataframe, target)
                if(verbose > 0):
                    print('Done...\n')
        print('=============================== END OF THE DATA PREPARATION ===============================\n')
    
    @classmethod
    def meta_prep(cls, target, transformer='gray', verbose=0):
        print('=========================== RUNNING THE PIPELINE RECOMMENDATION ===========================\n')
        if(verbose > 0):
            print('Checking the data and in which case fits better...')
            
        total_null = cls.dataframe.isna().sum().sum()
        num_columns, cat_columns, _, _, _ = separate(cls.dataframe, target)
        if(cat_columns is None):
            cat_columns = 0
        else:
            cat_null = cat_columns.isna().sum().sum()
            cat_columns = 1
        
        if(num_columns is None):
            num_columns = 0
        else:
            num_null = num_columns.isna().sum().sum()
            num_columns = 1
        
        if((cat_columns == 0) and (total_null == 0)): 
            case = 'metaprep/metamodels/metamodel_case1.pickle'
            if(verbose > 0):
                print('Case 1: Dataset without categorical data and null values.')

        elif((cat_columns == 0) and (num_null > 0)): 
            case = 'metaprep/metamodels/metamodel_case2.pickle'
            if(verbose > 0):
                print('Case 2: Dataset without categorical data but have null values in your set.')

        elif(((cat_columns == 1) and (total_null == 0)) or ((num_columns == 0) and (total_null == 0))):  
            case = 'metaprep/metamodels/metamodel_case3.pickle'
            if(verbose > 0):
                print('Case 3: Dataset without null values but have categorical data in your set.')            

        elif(((cat_columns == 1) and (cat_null > 0)) or ((num_columns == 0) and (cat_null > 0))):
            case = 'metaprep/metamodels/metamodel_case4.pickle'
            if(verbose > 0):
                print('Case 4: Dataset with categorical data and null values.')

        else:
            case = 'metaprep/metamodels/metamodel_case5.pickle'
            if(verbose > 0):
                print('Case 5: Dataset with categorical data but have null values only in numeric columns.')
            
        if(verbose > 0):
            print('Done...\n')
            print('Performing the data characterization...')
            
        meta_attr = data_caracterization(cls.dataframe, target, transformer)
        
        if(verbose > 0):
            print('Done...\n')
            print('Performing the pipeline recommendation...\n')
        
        with open(case, 'rb') as handle:
            metamodel = pickle.load(handle)
            
        recommend = metamodel.predict(meta_attr)

        recommend = recommend[0].split(".")
        for i in range(5):
            recommend[i] = list(recommend[i].split(","))
        
        if(verbose > 0):
            for i in range(5):
                print('Recommendation ', i, ': ', recommend[i])
            print('\nDone...\n')
        
        print('============================ END OF THE PIPELINE RECOMMENDATION ============================\n')
        return recommend
    
    @classmethod
    def unique_prep(cls, target, transformer='gray', verbose=0):
        print('=========================== RUNNING THE PIPELINE RECOMMENDATION ===========================\n')
        if(verbose > 0):
            print('Performing the data characterization...')
            
        meta_attr = data_caracterization(cls.dataframe, target, transformer)
        
        if(verbose > 0):
            print('Done...\n')
            print('Performing the pipeline recommendation...\n')
        
        with open('metamodels/metamodel_unique.pickle', 'rb') as handle:
            metamodel = pickle.load(handle)
        
        recommend = metamodel.predict(meta_attr)

        recommend = recommend[0].split(".")
        for i in range(5):
            recommend[i] = list(recommend[i].split(","))
            
        if(verbose > 0):
            for i in range(5):
                print('Recommendation ', i, ': ', recommend[i])
            print('\nDone...\n')
        
        print('============================ END OF THE PIPELINE RECOMMENDATION ============================\n')      
        return recommend
    
    @classmethod 
    def test(cls, target, algorithm, verbose=0):
        if(verbose > 0):
            print('Running the training...\n')
            verbose = 3
            
        x_train, x_test, y_train, y_test = train_test_split(cls.dataframe.drop([target], axis=1), 
                                                            cls.dataframe[target], test_size=0.33, random_state=42)
        if algorithm == 'svm':
            clf = SVC(verbose=verbose)
            model = clf.fit(x_train, y_train.ravel())
            if(verbose > 0):
                print('')
            print("Accuracy of the test with Support Vector Machine:", model.score(x_test,y_test.ravel()), '\n')
        
        if algorithm == 'rf':
            clf = RandomForestClassifier(verbose=verbose)
            model = clf.fit(x_train, y_train.ravel())
            if(verbose > 0):
                print('')
            print("Accuracy of the test with RandomForest:", model.score(x_test,y_test.ravel()), '\n')
            
    @classmethod
    def auto_prep(cls, target):
        total_null = cls.dataframe.isna().sum().sum()
        num_columns, cat_columns, _, _, _ = separate(cls.dataframe, target)
        if(cat_columns is None):
            cat_columns = 0
        else:
            cat_null = cat_columns.isna().sum().sum()
            cat_columns = 1
        
        if(num_columns is None):
            num_columns = 0
        else:
            num_null = num_columns.isna().sum().sum()
            num_columns = 1
        
        if((cat_columns == 0) and (total_null == 0)): 
            print('Dataset without categorical data and null values.')
            pipeline = ['no_preparation']
            print('No need data preparation.', '\n')

        elif((cat_columns == 0) and (num_null > 0)): 
            print('Dataset without categorical data but have null values in your set.')
            pipeline = ['imputation_median']
            print('Applied techniques: ', pipeline, '\n')

        elif(((cat_columns == 1) and (total_null == 0)) or ((num_columns == 0) and (total_null == 0))):  
            print('Dataset without null values but have categorical data in your set.')
            pipeline = ['label_encoder']
            print('Applied techniques: ', pipeline, '\n')

        elif(((cat_columns == 1) and (cat_null > 0)) or ((num_columns == 0) and (cat_null > 0))):
            print('Dataset with categorical data and null values.')
            pipeline = ['imputation_deletion_case', 'label_encoder', 'oversampling']
            print('Applied techniques: ', pipeline, '\n')

        else:
            print('Dataset with categorical data but have null values only in numeric columns.')
            pipeline = ['imputation_median', 'label_encoder']
            print('Applied techniques: ', pipeline, '\n')
        
        for stage in pipeline:
            if stage in cls.PIPELINE_OPTIONS:
                print("Stage --> ", stage)
                cls.dataframe = cls.PIPELINE_OPTIONS[stage](cls.dataframe, target)