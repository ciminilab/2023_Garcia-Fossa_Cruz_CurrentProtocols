"""
This script contains usefull functions used in the notebooks

@author: fefossa
"""

import pycytominer
import seaborn as sns
import matplotlib.pyplot as plt
import operator
import pandas as pd
import numpy as np
from utils.visualize_data import *

def pycytominer_operations(df, strata = 'Metadata_Compound_Concentration'):
    """"
    Normalize, select features and aggregate data to use this df on next steps
    df: dataframe with single cell information
    strata: Columns to groupby and aggregate rows based on the column or list of columns.
    """
    df_norm = pycytominer.normalize(df, method = 'mad_robustize', mad_robustize_epsilon = 0)
    df_selected = pycytominer.feature_select(df_norm, blocklist_file = 'blocklist_features.txt')
    print('Numbers of columns dropped after feature selection: ',df.shape[1] - df_selected.shape[1])
    df_ag = pycytominer.aggregate(df_selected, strata = strata)

    return df_ag

def corr_calculator(df_ag, strata = 'Metadata_Compound_Concentration'):
    """
    Take aggregated df, tranpose and create a correlation matrix based on all the features on dataset
    df_ag: aggregated dataset, which means each row is a well
    """
    df_T = df_ag.set_index(strata).T #transpose df making rows as the cols names using strata
    corr = df_T.corr() 

    return corr

def plot_corr(corr, fig_size = (15, 10)):
    """
    Take correlation df and plot a correlation matrix for visualization
    corr: correlation df
    """
    colormap = sns.color_palette("coolwarm", as_cmap=True)
    plt.figure(figsize = fig_size)
    plt.tick_params(axis='both', which='major', labelsize=15, labelbottom = False, bottom=False, top = False, labeltop=True)
    fig = sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap = colormap,
            annot=True)
    fig.set_xlabel('')
    plt.ylabel('Compound_Concentration \n', fontsize=16)

    return fig

def insert_corr(df, corr, corr_to = 'DMSO 0.00', sort_by = ['Metadata_Compound_Concentration']):
    """
    Insert correlation on the original df. This enables plotting images with the correlation values.
    
    """
    cmpd_cnc = df[sort_by].unique().tolist()
    #create dictionary based on which compound was defined at corr_to
    corr_dict = {}
    # print(cmpd_cnc)
    for i in range(len(cmpd_cnc)):
        n = 0
        for j in range(len(corr.index)):
            if corr_to in cmpd_cnc[i]:
                corr_dict[cmpd_cnc[n]] = corr.iloc[i][n]
                # print(i, j, n)
                n = n + 1
    print(corr_dict)
    #create Correlation_to col in df
    df['Correlation_to'] = 'NaN'
    for vals in range(len(df[sort_by].values)):
        for key,value in corr_dict.items():
            if key == df[sort_by].values[vals]:
                df['Correlation_to'].values[vals] = float(value)
    #format corr values 
    df['Correlation_to'] = df['Correlation_to'].map('{:.2f}'.format)
    
    return df 

def five_most_corr(df, corr, control = 'DMSO 0.00', add_moa = False):
    """
    Plot a table with the five compounds more closely correlated to the control choose by the user
    """
    cmpd_cnc = df['Metadata_Compound_Concentration'].unique().tolist()
    corr_dict = {}
    for i in range(len(cmpd_cnc)):
        n = 0
        for j in range(len(corr.index)):
            if cmpd_cnc[i] == control[0]:
                corr_dict[cmpd_cnc[n]] = corr.iloc[i][j]
                n = n+1
    #sort the dictionary on descending form
    sorted_dict = sorted(corr_dict.items(),key=operator.itemgetter(1),reverse=True)
    #create dataframe to store info and print five most correlated
    five_most = pd.DataFrame(sorted_dict[1:6], columns=['Compound', 'Correlation'])

    if add_moa:
        moas = []
        for cmp in range(len(five_most['Compound'].values)):
            for cmp2 in range(len(df['Metadata_Compound_Concentration'].values)):
                if five_most['Compound'].values[cmp] == df['Metadata_Compound_Concentration'].values[cmp2]:
                    moa = df['Metadata_moa'].values[cmp2]
                    moas.append(moa)
        moas = np.array(moas)
        moas = np.unique(moas)
        five_most['moa'] = moas

    print('Five compound most correlated to ', control[0])
   
    return five_most

def random_select(df, list_to_plot = ['DMSO'], sort_by = ['Metadata_Compound'], box_size = 150, correlation = False):
    """
    Split df to plot random cells. 
    To avoid cells to close to the border we use the following statements: 
    Nuclei_location_X and Nuclei_location_Y needs to be higher than the half_box_size (exlude Y and X low values) and lower than pic_size - half_box_size (exlude Y and X higher values)
    """
    n_cells = int(input('How many cells would you like to plot for each subgroup?'))
    selected = []
    for subgroup in list_to_plot:
        subset = df[df[sort_by] == subgroup] 
        sample = subset.reset_index(drop=True).sample(n_cells, replace = False).reset_index(drop=True) #choose n random cells
        for index in range(len(sample)):
        #the nuclei center needs to be higher than half box_size and lower than the image_size - half box_size
            if sample['Image_Width_OrigDNA'][index] - box_size/2 > sample['Nuclei_Location_Center_X'][index] >  0 + box_size/2 and 0 + box_size/2 < sample['Nuclei_Location_Center_Y'][index] < sample['Image_Width_OrigDNA'][index] - box_size/2:
                continue 
            else: 
                sample = subset.reset_index(drop=True).sample(n_cells, replace = False).reset_index(drop=True)
        selected.append(sample.values) 
    # create df from column names and selected cells
    df_selected_smp = pd.DataFrame(columns=df.columns, data=np.concatenate(selected)) 
    
    if correlation:
        df_selected_smp = df_selected_smp.sort_values(by=['Correlation_to'], ascending = False, ignore_index = True)

    return df_selected_smp

def representative_kmeans_select(df, list_to_plot = ['DMSO'], sort_by = ['Metadata_Compound'], box_size = 150, correlation = False):
    """
    
    """
    n_cells = int(input('How many cells would you like to plot for each group?'))
    selected = []
    cols = []
    for subgroup in list_to_plot:
        subset = df[df[sort_by] == subgroup]
        sample,_ = extract_single_cell_samples(subset,n_cells,cell_selection_method='representative')
        sample.drop(columns=['clusterLabels'], inplace=True)
        for index in range(len(sample)):
        #the nuclei center needs to be higher than half box_size and lower than the image_size - half box_size
            if sample['Image_Width_OrigDNA'][index] - box_size/2 > sample['Nuclei_Location_Center_X'][index] >  0 + box_size/2 and 0 + box_size/2 < sample['Nuclei_Location_Center_Y'][index] < sample['Image_Width_OrigDNA'][index] - box_size/2:
                continue 
            else: 
                sample,_ = extract_single_cell_samples(subset,n_cells,cell_selection_method='representative')
                sample.drop(columns=['clusterLabels'], inplace=True)
        selected.append(sample.values) 
    # create df from column names and selected cells
    df_selected_smp = pd.DataFrame(columns=df.columns, data=np.concatenate(selected)) 

    if correlation:
        df_selected_smp = df_selected_smp.sort_values(by=['Correlation_to'], ascending = False, ignore_index = True)
    
    return df_selected_smp

def representative_median_select(df, list_to_plot = ['DMSO'], sort_by = ['Metadata_Compound'], box_size = 150, correlation = False):
    """
    
    """
    n_cells = int(input('How many cells would you like to plot for each group?'))
    selected = []
    cols = []
    for subgroup in list_to_plot:
        subset = df[df[sort_by] == subgroup]
        sample,_ = extract_single_cell_samples(subset,n_cells,cell_selection_method='geometric_median')
        sample.drop(columns=['clusterLabels'], inplace=True)
        for index in range(len(sample)):
        #the nuclei center needs to be higher than half box_size and lower than the image_size - half box_size
            if sample['Image_Width_OrigDNA'][index] - box_size/2 > sample['Nuclei_Location_Center_X'][index] >  0 + box_size/2 and 0 + box_size/2 < sample['Nuclei_Location_Center_Y'][index] < sample['Image_Width_OrigDNA'][index] - box_size/2:
                continue 
            else: 
                sample,_ = extract_single_cell_samples(subset,n_cells,cell_selection_method='geometric_median')
                sample.drop(columns=['clusterLabels'], inplace=True)
        selected.append(sample.values) 
    # create df from column names and selected cells
    df_selected_smp = pd.DataFrame(columns=df.columns, data=np.concatenate(selected)) 

    if correlation:
        df_selected_smp = df_selected_smp.sort_values(by=['Correlation_to'], ascending = False, ignore_index = True)
    
    return df_selected_smp

def add_path(df, images_dir, channels = ["DNA","ER","RNA","AGP","Mito"], compressed = False, compressed_format = None):
    """
    Take path to images provided by the user and add to the dataframe

    If compressed, provide compressed_format for your images: it can be png, jpeg, jpg
    """
    df_random_comp = df.copy() #do a copy
        
    if compressed:
        for ch in channels:
            df_random_comp["PathName_Orig"+ch] = images_dir
            df_random_comp["FileName_Orig"+ch] = df_random_comp["Image_FileName_Orig"+ch].apply(lambda x: x.replace("tiff", compressed_format))

    if not compressed:
        for ch in channels:
            df_random_comp["PathName_Orig"+ch] = images_dir
            df_random_comp["FileName_Orig"+ch] = df_random_comp["Image_FileName_Orig"+ch]
    
    return df_random_comp

def plot_order(df_selected_smp, order = list, col_name = 'Metadata_Compound_Concentration'):
    """"
    User defines the order in which the pics will be ploted
    """
    dummy = pd.Series(order, name = col_name).to_frame()
    df_selected_smp = pd.merge(dummy, df_selected_smp, on = col_name, how = 'left')

    return df_selected_smp

def col_generator(df):
    """
    Create a new column containing information from compound + concentration of compounds
    """
    df['Metadata_Concentration'] = df['Metadata_Concentration'].map('{:.2f}'.format)
    df['Metadata_Compound_Concentration'] = df['Metadata_Compound'] + ' ' + df['Metadata_Concentration'].astype(str) #Join both columns
    print("Names of the compounds + concentration: ", df['Metadata_Compound_Concentration'].unique())

    return df
