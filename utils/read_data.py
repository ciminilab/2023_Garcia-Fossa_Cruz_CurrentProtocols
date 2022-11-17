"""
This script contains usefull functions used in the notebooks

@author: mhaghigh
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import os
import time

import pandas as pd
from sqlalchemy import create_engine
from functools import reduce


################################################################################
def extract_feature_names(df_input):
    """
    from the all df_input columns extract cell painting measurments 
    the measurments that should be used for analysis
    
    Inputs:
    df_input: dataframes with all the annotations available in the raw data
    
    Outputs: cp_features, cp_features_analysis
    
    """
    
    cp_features=df_input.columns[df_input.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    locFeature2beremoved=list(filter(lambda x: "_X" in x or "_Y" in x or "_x" in x or "_y" in x, cp_features)) 
    metadataFeature2beremoved=list(filter(lambda x: "etadata" in x , cp_features)) 
    # with open('./utils/blackListFeatures.pkl', 'rb') as f:
    #     blackListFeatures = pickle.load(f)
    
    
    cp_features_analysis=list(set(cp_features)-set(locFeature2beremoved)-set(metadataFeature2beremoved))

    return cp_features, cp_features_analysis

################################################################################
def handle_nans(df_input,cp_features):
    """
    from the all df_input columns extract cell painting measurments 
    the measurments that should be used for analysis
    
    Inputs:
    df_input: dataframes with all the annotations available in the raw data
    
    Outputs: cp_features, cp_features_analysis
    
    """
    
#     cp_features=df_input.columns[df_input.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()

    df_input=df_input.replace([np.inf, -np.inf], np.nan)
    
    null_vals_ratio=0.05; thrsh_std=0.0001;
    cols2remove_manyNulls=[i for i in cp_features if (df_input[i].isnull().sum(axis=0)/df_input.shape[0])\
                  >null_vals_ratio]   
    cols2remove_lowVars = df_input[cp_features].std()[df_input[cp_features].std() < thrsh_std].index.tolist()

    cols2removeCP = cols2remove_manyNulls + cols2remove_lowVars
#     print(cols2removeCP)

    cp_features_analysis = list(set(cp_features) - set(cols2removeCP))
    df_p_s=df_input.drop(cols2removeCP, axis=1);
    
    df_p_s[cp_features_analysis] = df_p_s[cp_features_analysis].interpolate()    
    
#     row_has_NaN = df_p_s[cp_features_analysis].isnull().any(axis=1)
#     print(row_has_NaN)
#     print(df_p_s[cp_features_analysis].dropna().shape,df_p_s[cp_features_analysis].shape)
#     df_p_s[cp_features_analysis] = df_p_s[cp_features_analysis].dropna() 
#     dataframe.fillna(0)
    
    return df_p_s, cp_features_analysis

    
    
    
    
    
