import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
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


def EvaluateClassification(Y_true, predicted, filename, normalize = True, 
                           save = False, path ='', imgname = 'img'):
    """
    Shows several metrics related to classification models to evaluate the performance
    of a model
    - Y_true (np.array/pd.Series): True labels.
    - predicted (np.array/pd.Series): Predicted labels made by the model.
    - filename (string): name of the saved log file.
    - normalize (bool): indicates if confusion matrix should be normalized.
    - save (bool): tells if the plot should be saved.
    - path (string): path where to save the figure. e.g.: 'images/'
    - imgname (string): name of the figure image file to be saved.
    
    Return (dict of floats): dict with all the metrics. They are: 'accuracy',
    'rocauc', 'mcc', 'macrof1' and 'microf1'.
    
    """
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'
    logResultsPath = os.getcwd()+sep+'results'+ sep + filename +'.txt'
    
    target_names = ['improcedente', 'procedente']
    
    metrics = dict()
    metrics['accuracy'] = accuracy_score(Y_true, predicted)
    metrics['rocauc'] = roc_auc_score(Y_true, predicted)
    metrics['mcc'] = matthews_corrcoef(Y_true, predicted)
    metrics['macrof1'] = f1_score(Y_true, predicted, average='macro')
    metrics['microf1'] = f1_score(Y_true, predicted, average='micro')
    
    results = "Accuracy: " + str(metrics['accuracy']) + "\n"
    results += "ROC AUC: " + str(metrics['rocauc']) + "\n"
    results += "Matthew Coefficient: " + str(metrics['mcc']) + "\n"
    results += "Macro F1-score: " + str(metrics['macrof1']) + "\n"
    print(results)
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
    
    
def PlotConfusionMatrix(cm, classes, normalize=False, title='Matriz de Confusão', 
                        cmap=plt.cm.Blues, save = False, path ='', imgname = 'img'):
    """
    This function prints and plots the confusion matrix.
    
    - cm (np.array): confusion matrix generated by scikit learn function
    confusion_matrix().
    - classes (list): list of unique labels.
    - normalize (bool): indicates if confusion matrix should be normalized.
    - title (string): Title of the plot image.
    - cmap (matplotlib.cm): Colormap used on the plot.
    - save (bool): tells if the plot should be saved.
    - path (string): path where to save the figure. e.g.: 'images/'
    - imgname (string): name of the figure image file to be saved.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão, sem normalização')
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cbar = ax.figure.colorbar(im, ax=ax, shrink = 0.7)
    tick_marks = np.arange(len(classes))
#    ax.set_xticks(tick_marks, classes, rotation=45)
#    ax.set_yticks(tick_marks, classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)    
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Classe Prevista', size=12)
    ax.set_ylabel('Classe Verdadeira', size=12)
    ax.titlesize = 13
    
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
#    plt.ylabel('Classe Verdadeira')
#    plt.xlabel('Classe Prevista')
    ax.labelsize = 12
    plt.show()
    
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