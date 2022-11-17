"""
This script contains usefull functions used in the notebooks

@author: mhaghigh and fefossa
"""
import asyncio
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import pickle
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import os
from functools import reduce
from sklearn.cluster import KMeans
from skimage import exposure
from skimage.transform import resize
import skimage.io
from utils.read_data import *


def extract_single_cell_samples(df_p_s,n_cells,cell_selection_method):
    """ 
    This function select cells based on input cell_selection_method
  
    Inputs: 
    ++ df_p_s   (pandas df) size --> (number of single cells)x(columns): 
    input dataframe contains single cells profiles as rows 
    
    ++ n_cells (dtype: int): number of cells to extract from the input dataframe
    
    ++ cell_selection_method (str): 
        - 'random' - generate n randomly selected cells
        - 'representative' - clusters the data and sample from the "closest to mean cluster"
        - 'geometric_median' - plots single sample than is the geometric median of samples 
           -> method 1 (hdmedians package): faster but can handle up to 800 rows without memory error
           -> method 2 (skfda package): slower but can handle up to 1500 rows without memory error
    
    Returns: 
    dff (pandas df): sampled input dataframe
  
    """    

    import hdmedians as hd
    from skfda import FDataGrid
    from skfda.exploratory.stats import geometric_median    
    cp_features, cp_features_analysis_0 =  extract_feature_names(df_p_s);
    df_p_s, cp_features_analysis = handle_nans(df_p_s,cp_features_analysis_0);
    
    
#     print("heloo")
#     print(cp_features)
    if cell_selection_method=='random':
        dff=df_p_s.reset_index(drop=True).sample(n = n_cells, replace = False).reset_index(drop=True)

    elif cell_selection_method=='representative': 
        df_p_s[cp_features_analysis] = df_p_s[cp_features_analysis].interpolate()
        if df_p_s.shape[0]>60:
            n_cells_in_each_cluster_unif=30
        else:
            n_cells_in_each_cluster_unif=int(df_p_s.shape[0]/5) 
            
        n_clusts=int(df_p_s.shape[0]/n_cells_in_each_cluster_unif) 
        kmeans = KMeans(n_clusters=n_clusts).fit(np.nan_to_num(df_p_s[cp_features_analysis].values))
        clusterLabels=kmeans.labels_
        df_p_s['clusterLabels']=clusterLabels;
        mean_clus=kmeans.predict(df_p_s[cp_features_analysis].mean().values[np.newaxis,])
        df_ps=df_p_s[df_p_s["clusterLabels"]==mean_clus[0]]
        dff=df_ps.reset_index(drop=True).sample(n = np.min([n_cells,df_ps.shape[0]]), replace = False).reset_index(drop=True)
#         print(dff)
#         print(cp_features_analysis)
    elif cell_selection_method=='geometric_median':    
    #     method 1
#         ps_arr=df_p_s[cp_features_analysis].astype(np.float32).values
        # ps_arr=df_p_s[cp_features_analysis].values
        # print(ps_arr)
        # gms=hd.medoid(ps_arr,axis=0)
        # gm_sample_ind=np.where(np.sum((ps_arr-gms),axis=1)==0)[0]
        # df_p_s_gm=df_p_s.loc[gm_sample_ind,:]
        # dff=pd.concat([df_p_s_gm,df_p_s_gm],ignore_index=True)

    # #     method 2
        ps_arr=df_p_s[cp_features_analysis].values
        print('1',ps_arr)
        X = FDataGrid(ps_arr)
        print('2',X)
        gms2 = np.squeeze(geometric_median(X).data_matrix)
        print('3',gms2)
        # gm2_sample_ind=np.where(np.sum((ps_arr-gms2),axis=1)==0)[0]
        gm2_sample_ind=np.array([np.argmin(np.sum(abs(ps_arr-gms2),axis=1))])
        print('4',gm2_sample_ind)
        df_p_s_gm2=df_p_s.loc[gm2_sample_ind,:]
        print('5',df_p_s_gm2)
        print(df_p_s_gm2.shape)
        dff=pd.concat([df_p_s_gm2,df_p_s_gm2],ignore_index=True)
        
    return dff,cp_features_analysis


def visualize_n_SingleCell(channels,sc_df,boxSize,title="",label=False,label_column=None,compressed=False,compressed_im_size=None, correlation=False, moa=False, rescale=False):
    """ 
    This function plots the single cells correspoding to the input single cell dataframe
  
    Inputs: 
    ++ sc_df   (pandas df) size --> (number of single cells)x(columns): 
    input dataframe contains single cells profiles as rows (make sure it has "Nuclei_Location_Center_X"or"Y" columns)
    
    ++ channels (dtype: list): list of channels to be displayed as columns of output image
           example: channels=['Mito','AGP','Brightfield','ER','DNA','Outline']
        * If Outline exist in the list of channels; function reads the outline image address from 
          "URL_CellOutlines" column of input dataframe, therefore, check that the addresses are correct
           before inputing them to the function, and if not, modify before input!
       
    ++ boxSize (int): Height or Width of the square bounding box
    
    Optional Inputs:
    ++ title (str)
    ++ compressed (bool) default is False, if set to True the next parameter is not optional anymore and should be provided
    ++ compressed_im_size (int), for example for lincs compressed is 1080
    ++ label (bool) default if False, if set to True the next parameter is not optional anymore and should be provided
    ++ label_column (str) provide a string with the name of the column the user want to use as the label
    
    Returns: 
    f (object): handle to the figure
  
    """
    compRatio = 1
    if compressed:
        
        original_im_size=sc_df['Image_Width_OrigDNA'].values[0]
        #         compressed_im_size=1080;
        compRatio=(compressed_im_size/original_im_size);
        boxSize = boxSize*compRatio ## compression change the boxSize ratio, so to look the same as non-compressed, this is necessary
        
#         sc_df['Nuclei_Location_Center_X']=sc_df['Nuclei_Location_Center_X']*compRatio
#         sc_df['Nuclei_Location_Center_Y']=sc_df['Nuclei_Location_Center_Y']*compRatio          

    
    halfBoxSize=int(boxSize/2);
#     print(channels)
    
    import skimage.io
    f, axarr = plt.subplots(sc_df.shape[0], len(channels),figsize=(len(channels)*2,sc_df.shape[0]*2));
    if len(title)>0:
        print(title)
        f.suptitle(title);
    
    f.subplots_adjust(hspace=0, wspace=0)


#     maxRanges={"DNA":8000,"RNA":6000,"Mito":6000,"ER":8000,"AGP":6000}
    for index in range(sc_df.shape[0]):
               
        xCenter=int(sc_df.loc[index,'Nuclei_Location_Center_X']*compRatio)
        yCenter=int(sc_df.loc[index,'Nuclei_Location_Center_Y']*compRatio)            
#         print(xCenter,yCenter)
        
        cpi=0;
        for c in channels:
            if c=='Outline':
                imPath=sc_df.loc[index,'Path_Outlines'];
                imD1=skimage.io.imread(imPath)
                
                if compressed:
                    imD1=skimage.transform.resize(imD,[compressed_im_size,compressed_im_size],mode='constant',preserve_range=True,order=0)
                    
                clim_max=imD1.max()
            else:
#                 ch_D=sc_df.loc[index,'Image_FileName_Orig'+c];
                ch_D=sc_df.loc[index,'FileName_Orig'+c];
#                 print(ch_D)
    #         imageDir=imDir+subjectID+' Mito_Morphology/'
#                 imageDir=sc_df.loc[index,'Image_PathName_Orig'+c]+'/'
                imageDir=sc_df.loc[index,'PathName_Orig'+c]+'/'
                imJoinPath=imageDir+ch_D
                imPath = os.path.abspath(imJoinPath)
            
                imD=skimage.io.imread(imPath)
                if rescale:
                    imD1=exposure.rescale_intensity(imD,in_range=(imD.min(),np.percentile(imD, 99.95)))
                    clim_max=imD1.max()
                else:
                    imD1 = imD
                
            imD_cropped=imD1[yCenter-halfBoxSize:yCenter+halfBoxSize,xCenter-halfBoxSize:xCenter+halfBoxSize]
#             axarr[index,cpi].imshow(imD,cmap='gray',clim=(0, maxRanges[c]));axarr[0,cpi].set_title(c);
            if rescale:
                axarr[index,cpi].imshow(imD_cropped,cmap='gray',clim=(0, clim_max));axarr[0,cpi].set_title(c);
            else:
                axarr[index,cpi].imshow(imD_cropped,cmap='gray');axarr[0,cpi].set_title(c);
            cpi+=1        
        
        ## add the labels in a way that when its a name with more than 2 words, it will split to the next line
        if label==True and correlation==True:
            corr = str(sc_df.loc[index,'Correlation_to'])
            imylabel=sc_df.loc[index,label_column] + ' Correlation_to_Control=' + corr #+'\n'+sc_df.loc[index,'Metadata_Compound']
            if ' ' in imylabel:
                newimylabel = imylabel.replace(' ', '\n')
                axarr[index,0].set_ylabel(newimylabel, rotation='horizontal', ha='right', va='center')
            else:
                axarr[index,0].set_ylabel(imylabel, rotation='horizontal', ha='right', va='center')
        else:
            pass
        
        if label==True and correlation==False:
#             corr = str(sc_df.loc[index,'Correlation_to'])
            imylabel=sc_df.loc[index,label_column] #+'\n'+sc_df.loc[index,'Metadata_Compound']
            if ' ' in imylabel:
                newimylabel = imylabel.replace(' ', '\n')
                axarr[index,0].set_ylabel(newimylabel, rotation='horizontal', ha='right', va='center')
            else:
                axarr[index,0].set_ylabel(imylabel, rotation='horizontal', ha='right', va='center')
        else:
            pass

        if moa==True:
            imylabel=sc_df.loc[index,'Metadata_moa'] 
            if ' ' in imylabel:
                newimylabel = imylabel.replace(' ', '\n')
                axarr[index,0].set_xlabel(newimylabel, rotation='horizontal', ha='right', va='center')
            else:
                axarr[index,0].set_xlabel(imylabel, rotation='horizontal', ha='right', va='center')
    

    for i in range(len(channels)):
        for j in range(sc_df.shape[0]):
            axarr[j,i].xaxis.set_major_locator(plt.NullLocator())
            axarr[j,i].yaxis.set_major_locator(plt.NullLocator())
            axarr[j,i].set_aspect('auto')
    
    return f

def visualize_image(channels,sc_df,title="",label=False,label_column=None,compressed=False,compressed_im_size=None,rescale=False):
    """ 
    This function plots the images correspoding to the chosen wells
  
    Inputs: 
    ++ sc_df   (pandas df) size --> (number of single cells)x(columns): 
    input dataframe contains single cells profiles as rows (make sure it has "Nuclei_Location_Center_X"or"Y" columns)
    
    ++ channels (dtype: list): list of channels to be displayed as columns of output image
           example: channels=['Mito','AGP','Brightfield','ER','DNA','Outline']
        * If Outline exist in the list of channels; function reads the outline image address from 
          "URL_CellOutlines" column of input dataframe, therefore, check that the addresses are correct
           before inputing them to the function, and if not, modify before input!
    ++ label: if label = True, it requires the label_column to add the respective names to each row
    
    ++ title
    
    Returns: 
    f (object): handle to the figure
  
    """
    import skimage.io
    import numpy as np
    import skimage.util
    f, axarr = plt.subplots(sc_df.shape[0], len(channels),figsize=(len(channels)*2,sc_df.shape[0]*2));
    if len(title)>0:
        print(title)
        f.suptitle(title);
    
    f.subplots_adjust(hspace=0, wspace=0)
    
    for index in range(sc_df.shape[0]):
        
        cpi=0;
        for c in channels:
            if c=='Outline':
                imPath=sc_df.loc[index,'Path_Outlines'];
                imD1=skimage.io.imread(imPath)
                
                if compressed:
                    imD1=skimage.transform.resize(imD,[compressed_im_size,compressed_im_size],mode='constant',preserve_range=True,order=0)
                    
                clim_max=imD1.max()
            else:
                ch_D=sc_df.loc[index,'FileName_Orig'+c];
                imageDir=sc_df.loc[index,'PathName_Orig'+c]+'/'
                imJoinPath=imageDir+ch_D
                imPath = os.path.abspath(imJoinPath)
            
                imD=skimage.io.imread(imPath)
                if rescale:
                    imD1=exposure.rescale_intensity(imD,in_range=(imD.min(),np.percentile(imD, 99.95)))
                    clim_max=imD1.max()
                else:
                    imD1 = imD
            if rescale:
                axarr[index,cpi].imshow(imD1,cmap='gray',clim=(0, clim_max));axarr[0,cpi].set_title(c);
            else:
                axarr[index,cpi].imshow(imD1,cmap='gray');axarr[0,cpi].set_title(c);
            cpi+=1        
        
        ## add the labels in a way that when its a name with more than 2 words, it will split to the next line
        if label:
            imylabel=sc_df.loc[index,label_column] #+'\n'+sc_df.loc[index,'Metadata_Compound']
            if ' ' in imylabel:
                newimylabel = imylabel.replace(' ', '\n')
                axarr[index,0].set_ylabel(newimylabel, rotation='horizontal', ha='right', va='center')
            else:
                axarr[index,0].set_ylabel(imylabel, rotation='horizontal', ha='right', va='center')
        else:
            pass

    for i in range(len(channels)):
        for j in range(sc_df.shape[0]):
            axarr[j,i].xaxis.set_major_locator(plt.NullLocator())
            axarr[j,i].yaxis.set_major_locator(plt.NullLocator())
            axarr[j,i].set_aspect('auto')
    
    return f
