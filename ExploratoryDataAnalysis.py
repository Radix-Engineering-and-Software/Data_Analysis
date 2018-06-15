import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from Methods import util

# Functions for Exploratory data analysis

def CalculateClassBalance(df, target = None, get=False):
    """
    Count the number of occurence of each class.
    
    - df (pandas dataframe): dataframe containing the data.
    - target (string/number): Name of the target column (feature). If none,
    the target feature is considered to be the last one.
    - get (bool): if True, the function returns a dict with the classes balance
    
    Return (dict)
    """
    if target is None:
        labels = df.iloc[:,-1]
    else:
        labels = df[target]

    # Calculate number of occurences for each class and total ocurrences
    balance = dict()
    total = 0
    for label in labels.unique():
        balance[label] = (labels == label).sum()
        total += balance[label]
    
    # Transform values to percentage
    valueInPercent = np.array(list(balance.values())) / total
    valueInPercent = valueInPercent * 100
    balanceInPercent = dict(zip(balance.keys(), valueInPercent))
    
    # Show values
    print('Number of ocurrences for each class:')
    for label in balance.keys():
        print('Class %s: %.2f%% (%d)' %(label, balanceInPercent[label], balance[label]))
        
    if (get):
       return balance 
        
def NumberOfNansInEachColumns(df):
    """
    Count the number of Not a Number (NaNs) on each column (feature).
    
    - df (pandas dataframe): dataframe containing the data.
    """
    total = len(df)
    dfNans = df.isnull().sum()
    dfNansInPercent = (dfNans * 100) / total
    
    print('Number of Nans in each column')
    for column in dfNans.index:        
        print('%s : %.2f %% (%d)' %(column, dfNansInPercent[column], dfNans[column]))
        
def UniqueValuesOnEachColumn(df, showValues=False):
    """
    Show how many unique values are there on each column(feature).
    
    - df (pandas dataframe/ pandas Series): Data.
    - showValues (bool): tells if each unique value of each column show be showed.
    
    Return (dict): Dicionary wit the number of unique values for each column
    """
    uniqueValues = dict()
    print('Frequency of each unique value:')
    if isinstance(df, pd.Series):
        uniqueValues[df.name] = len(df.unique())
        print('- %s : %d unique values' %(df.name, len(df.unique())))        
        if showValues:
            total = len(df)
            occurrences = dict()
            occurrencesInPercent = dict()
            for value in df.unique():
                occurrences[value] = (df == value).sum()
                occurrencesInPercent[value] = (occurrences[value] * 100.0) / total
                print("\t%s : %.2f%% (%d)" %(str(value), occurrencesInPercent[value], occurrences[value]))
    else: # df is assumed to be pandas dataframe
        for column in df.columns:
            uniqueValues[column] = len(df[column].unique())
            print('- %s : %d unique values' %(column, len(df[column].unique())))
            if showValues:
                total = len(df)
                occurrences = dict()
                occurrencesInPercent = dict()
                for value in df[column].unique():
                    occurrences[value] = (df == value).sum()
                    occurrencesInPercent[value] = (occurrences[value] * 100.0) / total
                    print("\t%s : %.2f%% (%d)" %(str(value), occurrencesInPercent[value], occurrences[value]))            
    
    return uniqueValues

def PlotCodeFrequency(codes, labels, save = False, path ='', filename = 'img'):
    """
    Plot the frequency of each code on the dataset and also which ones are DI or DP.
    
    - codes (pandas Series, list, numpy array): codes of each sample.
    - labels (pandas Series, list, numpy array): labels of each sample.
    - save (bool): tells if the plot should be saved.
    - path (string): path where to save the figure.
    - filename (string): name of the figure image file to be saved.
    """
    DI = 0
    DP = 1
    
    df=pd.DataFrame()
    df['labels'] = labels
    df['codes'] = codes
    
    freq_di = dict()
    freq_dp = dict()
    
    total = len(df)
    N=0
    toPercentage = lambda value,total: (value * 100.0) / total
    codigosExistentes = list(df['codes'].unique())
    codigosExistentes.sort()
    for c in codigosExistentes:
        freq_di[c] = toPercentage(((df['codes'] == c) & (df['labels'] == DI)).sum(), total)
        freq_dp[c] = toPercentage(((df['codes'] == c) & (df['labels'] == DP)).sum(), total)
        N += 1
        
    ind = np.arange(N)
    width = 0.5
    
    fig = plt.figure(figsize=(11, 5))
    dp_bar = plt.bar(ind, list(freq_dp.values()), width, figure=fig)
    di_bar = plt.bar(ind, list(freq_di.values()), width, bottom=list(freq_dp.values()), figure=fig)
    
    minorTicks = MultipleLocator(1)
    
    plt.ylabel('Porcentagem (%)')
    plt.xlabel('Códigos')
    plt.title('Frequência de cada código no dataset')
    plt.xticks(ind, tuple(freq_di.keys()))
    plt.yticks(np.arange(0, 25, 5))
    plt.axes().yaxis.set_minor_locator(minorTicks)
    plt.legend((di_bar[0], dp_bar[0]), ('DI', 'DP'))
    plt.grid(True, which='both', axis='y')
    
    plt.show()
    
    if(save):
        util.CheckAndCreatePath(path)
        util.SaveFigure(fig, path, filename)
    