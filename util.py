import numpy as np
import pandas as pd
import logging
import json
import os
import sys

def SaveResultsIntoFile(results, path):
    """
    Saves log text file to a specified path.
    
    - results (string): log lext. 
    - path (string): path where log file should be saved at.
    """
    file = open(path,"w") 
    file.write(results) 
    file.close()    

def CheckAndCreatePath(path):
    """
    Check if a path exists, if it doesn't, then create the path.
    
    - path (string): path to be checked and created.
    """
    if (path) and (not os.path.isdir(path)):
        os.makedirs(path)

def SaveFigure(figure, path, filename):
    """
    Save a figure of a matplotlib plot.
    
    - figure (matplotlib figure): figure to be saved.
    - path (string): path where to save the figure.
    - filename (string): name of the figure to be saved.
    """
    if (path != ''):
        is_windows = sys.platform.startswith('win')
        sep = '\\'
    
        if is_windows == False:
            sep = '/'
            
        if (path[-1] == '/' or path[-1]=='\\'):
            path = path[:-1]
            
        figure.savefig(path + sep + filename + '.png', bbox_inches='tight')
    else:
        figure.savefig(filename + '.png', bbox_inches='tight')
        
def ConcatenateDataframes(left, right):
    """
    Concatenate two pandas dataframe, avoiding bug from pandas.concatenate()
    function, where one of the dataframes become full of NaNs.
    
    - left (pandas dataframe): Left side dataframe.
    - right (pandas dataframe): Left side dataframe.
    
    Return (pandas dataframe): resulting concatenated dataframe
    """
    l1 = left.values.tolist()
    l2 = right.values.tolist()
    for i in range(len(l1)):
        l1[i].extend(l2[i])
    
    return pd.DataFrame(l1, columns = left.columns.tolist() + right.columns.tolist())

def InsertColumnsOnADataframePosition(df1, df2, pos):
    """
    Insert columns from df2 to df1 starting on a specified position.
    - df1 (pd.Dataframe): dataframe where the columns will be inserted.
    - df2 (pd.Dataframe): dataframe with the columns to be inserted.
    - pos (int): position where the columns should be inserted.
    
    Return (pd.Dataframe): Resulting combined Dataframe.
    """
    
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    
    cols = cols1[:pos]+cols2+cols1[pos:]
    
    df = pd.concat([df1,df2], axis=1)
    df = df[cols]
        
    return df
    
def log(name, logFile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(logFile)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(streamHandler)
    
    
def RecodeEmptyCells(dataframe):
    dataframe = dataframe.replace(r'\s+', np.nan, regex=True)
    dataframe = dataframe.fillna('0')
    return dataframe

def JsonToDataframe(argument):
    data = {}
    json_data = json.loads(argument)
    
    for obj in json_data:
        for col, val in obj.items():
            try:
                data[col].append(val)
            except:
                data.update({col: [val]})

    return pd.DataFrame(data)

def IsNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        pass    
    return False


def RemoveSpacesAndNaN(arr, col):
    for i in range(len(arr[:,col])):
        try:
            arr[i,col] = arr[i,col].strip()
        except:
            arr[i,col] = '-1'
    return arr


def RemoveLetters(arr, col, letters):
    for i in range(len(arr[:,col])):
        try:
            arr[i,col] = arr[i,col].strip(letters)
            if IsNumber(arr[i,col]) == False:
                arr[i,col] = 0
        except:
            pass
    return arr


def checker(arr):
    index=0
    for i in arr:
        try:
            if not np.isfinite(i):
                print(index, i)
        except:
            pass
        index +=1