from operator import length_hint
from socket import RDS_GET_MR_FOR_DEST
import pandas as pd
import numpy as np
from regex import D
from scripts.data_visualization import Data_Viz;

class DataCleaner:
    """
    Class for cleaning our given data
    """
    def __init__(self) -> None:
        self.summar=Data_Viz()
        
    '''
    removing columing with missing values exceeding the limit set
    ''' 
    def remove_unlimited_missing(self, df, limit):
        temp = self.summar.summ_colums(df)
        rm_list = []
        for i in range(temp.shape[0]):
            if(temp.iloc[i,2] > limit):
                rm_list.append(temp.iloc[i,0])
        rm_df = df.drop(rm_list, axis=1)
        return rm_df
    '''
    filling missing values with mode
    '''
    def fill_missing_by_mode(self, df, cols=None):
        fill_by_mode = []
        temp = self.summar.summ_colums(df)
        if(cols == None):
            for i in range(temp.shape[0]):
                if(temp.iloc[i,3] == "object"):
                    fill_by_mode.append(temp.iloc[i,0])
        else:
            for col in cols:
                fill_by_mode.append(col)
                
        for col in fill_by_mode:
          df[col] =df[col].fillna(df[col].mode[0])
        
        return df

    '''
    filling missing values with mean
    '''
    def fill_missing_by_mean(self, df, cols=None):
        fill_by_mean = []
        temp = self.summar.summ_columns(df)
        
        if(cols == None):
            for i in range(temp.shape[0]):
                if(temp.iloc[i,3] == "object"):
                    fill_by_mean.append(temp.iloc[i,0])
        else:
            for col in cols:
                fill_by_mean.append(col)
                
        for col in fill_by_mean:
          df[col] =df[col].fillna(df[col].mean())
        
        return df
        
    '''
    filling missing values with median
    '''
    def fill_missing_by_median(self, df, cols=None):
        fill_by_median = []
        temp = self.summar.summ_columns(df)
        
        if(cols == None):
            for i in range(temp.shape[0]):
                if(temp.iloc[i,3] == "float64" or temp.iloc[i,3] == "int64"):
                    fill_by_median.append(temp.iloc[i,0])
        else:
            for col in cols:
                fill_by_median.append(col)
                
        for col in fill_by_median:
          df[col] =df[col].fillna(df[col].medina())
        
        return df
        
    def fill_missing_ffill(self, df, cols):
        """
        filling missing values by value from next rows
        """
        for col in cols:
            df[col] = df[col].fillna(method='ffill')
        return df
        
    def fill_missing_bfill(self, df, cols):
        """
        filling missing values by value from previous rows
        """
        for col in cols:
            df[col] = df[col].fillna(method='bfill')
        return df
        
    def fill_outliers(self, df, cols):
        df_temp =df.copy(deep=True)
        for col in cols:
            Q1 = df_temp[col].quantile(0.25)
            Q2 = df_temp[col].quantile(0.75)
            IQR = Q2 - Q1
            
            length=df_temp.shape[0]
            for index in range(length):
                if(df_temp.loc[index,col] >= (Q2+1.5*IQR)):
                    df_temp.loc[index,col] = np.nan
                    
            df_temp = self.fill_missing_by_median(df_temp, cols)
        return df_temp 
        
    def remove_outliers(self, df,cols):
        df_temp = df.copy(deep=True)
        for col in cols:
            Q1 = df_temp[col].quantile(0.25)
            
            Q2 = df_temp[col].quantile(0.75)
            IQR = Q2 - Q1
            rm_list =[]
            length=df_temp.shape[0]
            for index in range(length):
                if(df_temp.loc[index,col] >= (Q2+1.5*IQR)):
                    rm_list.append(index)
                    
            df_temp.drop(df_temp, inplace = True)
            
        return df_temp
        
        
    """
    removing specified columns from dataframe
    """   
    def remove_cols(self, df, cols, keep=False):
        if(keep):
            rm_df = df.loc[:,cols]
        else:
            rm_df = df.drop(cols, axis=1)
        return rm_df