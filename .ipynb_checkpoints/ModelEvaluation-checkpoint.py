import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from statsmodels.stats.proportion import proportion_confint
from math import sqrt
from scipy import stats
import os
import sys
import matplotlib.pyplot as plt
import itertools
import xgboost as xgb
from Methods import util



def EvaluateRegression(Y_true, predicted, filename, analyze_minor_class=False):
    """
    Shows several metrics related to regression models to evaluate the performance
    of a model
    - Y_true (np.array/pd.Series): True labels.
    - predicted (np.array/pd.Series): Predicted labels made by the model.
    - filename (string): name of the saved log file.
    - analyze_minor_class (bool): indicates if metrics realted to the minor class
    should be shown.
    """
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'
    logResultsPath = os.getcwd()+sep+'results'+ sep + filename +'.txt'

    results = "MAE: " + str(mean_absolute_error(Y_true, predicted)) + "\n"
    results += "RMSE: " + str(sqrt(mean_squared_error(Y_true, predicted))) + "\n"
    results += "Expl. variance score: " + str(explained_variance_score(Y_true, predicted)) + "\n"
    results += "Coefficient of determination (R^2): " + str(r2_score(Y_true, predicted)) + "\n"
    results += "Log loss: " + str(log_loss(Y_true, predicted))
    print(results)
    
    ## Save training results
    util.SaveResultsIntoFile(results, logResultsPath)

    if analyze_minor_class:
        scores_minor_class = [predicted[i] for i in range(len(predicted)) if Y_true[i]==0]
        print("Total of test instances:", len(Y_true))
        print("Total of test instances from minor class:", len(scores_minor_class))
        print("Mean of minor class:", np.mean(scores_minor_class))
        print("Mode of minor class:", stats.mode(scores_minor_class))
        print("Worst score of minor class:", np.min(scores_minor_class))
        print("Best score of minor class:", np.max(scores_minor_class))
        
        print("printing the regression results ...")
        for score in scores_minor_class:
            print(score)



    PlotConfusionMatrix(confusion_matrix(Y_true, predicted), target_names, normalize=normalize,
                        save = save, path = path, imgname = imgname)
    
    # Classification Report
    print(classification_report(Y_true, predicted, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(Y_true, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    results += "Normalized Confusion Matrix:\n"
    results += "| " + str(round(cm[0,0], 2)) + "  " + str(round(cm[0,1], 2)) + " |\n"
    results += "| " + str(round(cm[1,0], 2)) + "  " + str(round(cm[1,1], 2)) + " |"
    
    ## Save training results
    util.SaveResultsIntoFile(results, logResultsPath)
    
    return metrics
    
    
def CalculateConfusionMatrixCI(cm, alpha=0.5):
    """
    Calculates confidence intervals for confusion matrix using Wilson score interval method.
    
    Parameters
    ----------
    cm: numpy.array
        Confusion Matrix.
    alpha: float
        Significance level.
        
    Returns
    -------
    cm_ci: numpy.array
        Calculated donfidence interval matrix
    """
    cm_ci = np.copy(cm).astype(float)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            nobs = np.sum(cm[i])
            count = cm[i, j]
            interval_limits = proportion_confint(count=count, nobs=nobs, alpha=0.05, method='wilson')
            cm_ci[i, j] = (interval_limits[1] - interval_limits[0])/2
    
    return cm_ci

    
def EvaluateClassification(Y_true, predicted, normalize=True, target_names='', filename='log',
                           save=False, path='', imgname='img', calculate_ic=False):
    """
    Calculates metrics related to classification models to evaluate its the performance.
    
    Parameters
    ----------
    cm : numpy.array
        Matriz de confusão.
    Y_true : numpy.array, pandas.Series
        True labels.
    predicted : numpy.array, pandas.Series
        Predicted labels made by the model.
    filename : str, optional
        Name of the saved log file.
    normalize : bool, optional
        Indicates if confusion matrix should be normalized.
    save : bool, optional
        Indicates if the plot should be saved.
    path : str, optional
        Path where to save the figure. e.g.: 'images/'
    imgname : str, optional
        Name of the figure image file to be saved.
    calculate_ic : bool, optional
        Indicates if the confidence intervals should be calculated with 0.05 significance level. Ignored if normalize is False.
    
    Returns
    -------
    metrics : Dict[str, float]
        Calculated metrics ('accuracy', 'rocauc', 'mcc', 'macrof1' and 'microf1').
    
    """
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'
    logResultsPath = os.getcwd()+sep+'results'+ sep + filename +'.txt'

    if target_names =='':
        target_names = ['DP', 'DI']
    else:
        target_names = target_names        
    
    
    conf_matrix = confusion_matrix(Y_true, predicted)
    if (normalize and calculate_ic):
        cm_confidence_interval = CalculateConfusionMatrixCI(conf_matrix)
    else:
        cm_confidence_interval = None
    
    metrics = dict()
    metrics['accuracy'] = accuracy_score(Y_true, predicted)
    metrics['rocauc'] = roc_auc_score(Y_true, predicted)
    metrics['mcc'] = matthews_corrcoef(Y_true, predicted)
    metrics['macrof1'] = f1_score(Y_true, predicted, average='macro')
    metrics['microf1'] = f1_score(Y_true, predicted, average='micro')
    metrics['confusion_matrix'] = conf_matrix
    metrics['confusion_matrix_ci'] = cm_confidence_interval
    
       
    results = "Accuracy: " + str(metrics['accuracy']) + "\n"
    results += "ROC AUC: " + str(metrics['rocauc']) + "\n"
    results += "Matthew Coefficient: " + str(metrics['mcc']) + "\n"
    results += "Macro F1-score: " + str(metrics['macrof1']) + "\n"
          
    print(results)
    PlotConfusionMatrix(conf_matrix, cm_confidence_interval, target_names, normalize=normalize,
                        save = save, path = path, imgname = imgname)
    
    results += "Confusion Matrix:\n"
    results += f"| {conf_matrix[0,0]:.2f}  {conf_matrix[0,1]:.2f} |\n"
    results += f"| {conf_matrix[1,0]:.2f}  {conf_matrix[1,1]:.2f} |\n"
    if (normalize and calculate_ic):
        results += "Confusion Matrix Confidence Interval:\n"
        results += f"| {cm_confidence_interval[0,0]:.2f}  {cm_confidence_interval[0,1]:.2f} |\n"
        results += f"| {cm_confidence_interval[1,0]:.2f}  {cm_confidence_interval[1,1]:.2f} |\n"
    
    ## Save training results
    util.SaveResultsIntoFile(results, logResultsPath)
    
    # Classification Report
    print(classification_report(Y_true, predicted, target_names=target_names))
    
    return metrics

def PlotConfusionMatrix(cm, cm_confidence_interval,  classes, normalize=False,
                        title='Matriz de Confusão', cmap=plt.cm.Blues,
                        save=False, path='', imgname='matriz_de_confusão'):
    """Gera e mostra a matriz de confusão.

    Parameters
    ----------
    cm : numpy.array
        Matriz de confusão.
    classes : list
        Lista com o nome de cada classe dos rótulos. Exemplo: ['DI', 'DP'].
    normalize : bool, default False
        Se verdadeiro, a matriz de confusão será normalizada.
    title : string, default 'Matriz de Confusão'
        Título da imagem da matriz de confusão.
    cmap : matplotlib.pyplot.cm, default matplotlib.pyplot.cm.Blues
        Colormap usado na matriz.
    save : bool, default False
        Se verdadeiro, a imagem da matriz de confusão será salva.
    path : str
        Diretório onde a imagem da matriz de confusão será salva.
    imgname : str, default 'img'
        Nome da imagem da matriz de confusão que será salva.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão, sem normalização')

    # formatted confusion matrix
    cm_format = np.copy(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                if cm_confidence_interval is not None:
                    cm_format[i,j] = f'{cm[i,j]:.2f} ' + u"\u00B1" + f' {cm_confidence_interval[i,j]:.4f}'
                else:
                    cm_format[i,j] = f'{cm[i,j]:.2f}'
            else:
                cm_format[i,j] = f'{cm[i,j]:.0f}'
                

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, aspect='equal', interpolation='nearest', cmap=cmap)
    plt.title(title)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Classe Prevista', size=12)
    ax.set_ylabel('Classe Verdadeira', size=12)
    ax.titlesize = 13

    thresh = (cm.max() + cm.min()) / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm_format[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)

    plt.tight_layout()
    ax.labelsize = 12
    plt.grid(False)
    plt.show(block=False)

    if(save):
        util.CheckAndCreatePath(path)
        util.SaveFigure(fig, path, imgname)
        

    
def AccuracyByEventCode(code, codes, labels, predictions):
    """
    Given the code of an event, it tells how many of that code the model gets right.
    
    - code (number): 'Causa Codigo' column code.
    - codes (numpy.array / pd.Series/ list): code of each event.
    - labels (numpy.array / pd.Series/ list): True labels of each event.
    - predictions (numpy.array / pd.Series/ list): Model predictions of each event.    
    """
    DI = 0
    DP = 1
    
    df = pd.DataFrame()
    df['codes'] = codes
    df['labels'] = labels
    df['predictions'] = predictions
        
    total = len(df)
    
    codeSelection = df[df['codes'] == code]
    totalSelection = len(codeSelection)
    
    # Model Accuracy
    rightPredictions = (codeSelection['labels'] == codeSelection['predictions']).sum()
    wrongPredictions = len(codeSelection) - rightPredictions
    
    # Frequency
    freq = (df['codes'] == code).sum()
    numberOfDI = (df['labels'] == DI).sum()
    numberOfDP = (df['labels'] == DP).sum()
    freqDI = (codeSelection['labels'] == DI).sum()
    freqDP = (codeSelection['labels'] == DP).sum()
    
    # Original code distribution
    label_DI = (codeSelection['labels'] == DI).sum()
    label_DP = totalSelection - label_DI
    
    # Predictions distribution
    pred_DI = (codeSelection['predictions'] == DI).sum()
    pred_DP = totalSelection - pred_DI
    
    toPercentage = lambda value,total: (value * 100.0) / total 
    
    # print information about the event code
    print('Code %d' %code)
    
    if (total > 0):
        # Frequency
        print('Frquency on the dataset: %.2f %% (%.2f %% of DI and %.2f %% of DP)' 
              %(toPercentage(freq,total), toPercentage(freqDI, numberOfDI), toPercentage(freqDP,numberOfDP)))
        
    if(totalSelection > 0):
         # Model Accuracy
        print('\nPercentage of right predictions: %.2f %%' %toPercentage(rightPredictions, totalSelection))
        print('Percentage of wrong predictions: %.2f %%' %toPercentage(wrongPredictions, totalSelection))
        
        # Original code distribution
        print('\nCode distritution on original dataset (labels):')
        print('Deslocamento Improcedente: %.2f %%' %toPercentage(label_DI, totalSelection))
        print('Deslocamento Procedente: %.2f %%' %toPercentage(label_DP, totalSelection))
        
        # Predictions distribution
        if( toPercentage(freqDI, numberOfDI) > 0 and toPercentage(freqDP,numberOfDP) > 0):
            print('\nCode distritution according to the model (predictions):')
            print('Deslocamento Improcedente: %.2f %%' %toPercentage(pred_DI, totalSelection))
            print('Deslocamento Procedente: %.2f %%' %toPercentage(pred_DP, totalSelection))

def PlotAccuracyOfEachEventCode(codes, labels, prediction, save = False, path ='', filename = 'img'):
    """
    Plots the model accuracy for each event code.
    
    - codes (pandas Series, list, numpy array): codes of each sample.
    - labels (pandas Series, list, numpy array): labels of each sample. 
    - prediction (pandas Series, list, numpy array): predictions of each sample.
    - save (bool): tells if the plot should be saved.
    - path (string): path where to save the figure. e.g.: 'images/'
    - filename (string): name of the figure image file to be saved.
    """
    
    df=pd.DataFrame()
    df['labels'] = labels
    df['codes'] = codes
    df['prediction'] = prediction
    
    cods = df['codes'].unique()
    cods = np.sort(cods)
    cods = cods[np.invert(np.isnan(cods))]
    cods = cods.astype(int)
    
    toPercentage = lambda value,total: (value * 100.0) / total
    accuracy = list()
    for cod in cods: 
        right = ((df['prediction'] == df['labels']) & (df['codes'] == cod)).sum()
        total = (df['codes'] == cod).sum()
        percentage = toPercentage(right, total)
        accuracy.append(percentage)
        
    # Plotar graficos
    fig, ax = plt.subplots(figsize=(11, 5))
    xticks = list(range(0, len(cods)))
    yticks = list(range(0,101,10))
    
    # show the figure, but do not block
    
    plt.bar(xticks, accuracy,figure = fig, align='center', width=0.3)
    ax.tick_params(axis='y', gridOn = True)
    ax.set_xticks(xticks)
    ax.set_xticklabels(cods)
    ax.set_yticks(yticks)
    ax.set_ylim([0, 100])
    ax.set_ylabel('Acurácia (%)')
    ax.set_xlabel('Codigo')
    ax.set_title('Acurácia do modelo para cada código')
    
    plt.show(block=False)
    
    if(save):
        util.CheckAndCreatePath(path)
        util.SaveFigure(fig, path, filename)
    
def PlotFeatureImportanceXGBoost(model, save = False, path ='', filename = 'img'):
    """
    Plots the importance of each feature from a XGBoost model.
    
    - model (XGBoost model): model.
    - save (bool): tells if the plot should be saved.
    - path (string): path where to save the figure.
    - filename (string): name of the figure image file to be saved.
    """
    fig, ax = plt.subplots(figsize=(6,7)) 
    xgb.plot_importance(model, ax=ax)
    plt.show()
    
    if(save):
        util.CheckAndCreatePath(path)
        util.SaveFigure(fig, path, filename)