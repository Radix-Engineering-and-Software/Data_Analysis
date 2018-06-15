from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import utils
from reportlab.platypus import Table, TableStyle, Image
from Methods import ExploratoryDataAnalysis as EDA
from Methods import ModelEvaluation
from Methods import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def InsertSection(c, dy, texto, cursor):
    c.saveState()
    c.translate(0, cursor+dy)
    
    # Draw Rectangle
    c.setFillColorRGB(0.74,0.84,0.93)
    c.rect(0, 0, WIDTH - 2*MARGIN_X, 1*cm, fill=1, stroke=0)
    
    # Write Title
    x = (WIDTH - 2*MARGIN_X)/2
    c.setFont("Helvetica", 14)
    c.setFillColorRGB(0,0,0)
    c.drawCentredString(x, 0.3*cm, texto)
    
    c.restoreState()
    cursor -= dy + 1*cm
    cursor = SkipLine(cursor,0)
    
    return cursor

def InsertField(table, c, cursor):
    lins, cols = np.array(table).shape
    
    style = TableStyle([('LINEBELOW', (0,0), (-1,-1), 1, colors.black),
                        ('FACE', (0,0), (-1,-1), 'Helvetica'),
                        ('SIZE', (0,0), (-1,-1), 12),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('TEXTCOLOR',(1,0),(1,-1),colors.Color(0.4,0.4,0.4)),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT')])
    
    for l in range(lins):
        if(l%2 == 0):
            style.add('BACKGROUND', (0,l), (-1,l), colors.Color(0.9,0.9,0.9))
            
    t = Table(table, style=style, rowHeights = LIN_HEIGHT)
    
    sizex, sizey = t.wrap(WIDTH, HEIGHT)    
    cursor -= sizey
    t.drawOn(c, 0, cursor)
    cursor = SkipLine(cursor, n=2)
    return cursor

def InsertImage(path, c, width, cursor):
    # Adjust aspect ratio
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    
    img = Image(path, width=width, height=(width * aspect))     
    img.drawOn(c, 0, cursor)
    cursor -= img.imageHeight
    
    return cursor

def SkipLine(cursor, n=1):
    cursor -= n*LIN_HEIGHT
    return cursor

def Generate(model, modelName, codes, X_train, Y_train, X_test, Y_test, path='Reports/'):
    # Generate needed variables
    X = pd.concat([X_train, X_test])
    Y = pd.concat([Y_train, Y_test])
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)
    
    data = datetime.now()
    nomeArquivo = 'relatorio'+ data.strftime('_%d_%m_%Y-%H_%M_%S') + '.pdf'
    titulo = 'Relatório do Modelo'
    dataString = data.strftime('%d/%m/%Y %H:%M:%S')
    
    colunas = list(X_train.columns)
    ModelEvaluation.PlotFeatureImportanceXGBoost(model, save=True, path='images',
                                                 filename='feature_importance')
    
    uniqueValues = EDA.UniqueValuesOnEachColumn(X)
    ClassBalance = EDA.CalculateClassBalance(Y.to_frame(), get=True)
    EDA.PlotCodeFrequency(codes, Y, save = True, path ='images', 
                          filename = 'code_frequency')
    
    metrics_test = ModelEvaluation.EvaluateClassification(Y_test, predictions_test, 
                                                     'classifier_scores_v3', 
                                                     save=True, 
                                                     path='images',
                                                     imgname='confusion_matrix_test')
    ModelEvaluation.PlotAccuracyOfEachEventCode(codes[len(Y_train):], Y_test, 
                                                      predictions_test, save = True, 
                                                      path='images',
                                                      filename = 'accuracy_by_code_test')
    
    metrics_train = ModelEvaluation.EvaluateClassification(Y_train, predictions_train, 
                                                     'classifier_scores_v3', 
                                                     save=True, 
                                                     path='images',
                                                     imgname='confusion_matrix_train')
    ModelEvaluation.PlotAccuracyOfEachEventCode(codes[:len(Y_train)], Y_train, 
                                                      predictions_train, save = True, 
                                                      path='images',
                                                      filename = 'accuracy_by_code_train')
    
    # Create document
    
    # if directory doesnt exist, create it
    util.CheckAndCreatePath(path)
    
    c = canvas.Canvas(path + nomeArquivo, pagesize=A4)
    global WIDTH
    global HEIGHT
    global MARGIN_Y
    global MARGIN_X
    global LIN_HEIGHT
    
    WIDTH, HEIGHT = A4
    MARGIN_Y = 2.54*cm
    MARGIN_X = 1.5*cm
    LIN_HEIGHT = 0.8*cm
    
    c.translate(MARGIN_X, 0)
    cursor = HEIGHT - MARGIN_Y
    
    ##########
    # Page 1 #
    ##########
    
    # Title
    c.saveState()
    c.translate(0, cursor)
    c.setFont("Helvetica", 20)
    c.setFillColorRGB(0,0,0)

    x = (WIDTH - 2*MARGIN_X)/2
    c.drawCentredString(x, 0, titulo)
    cursor -= 1.3*cm
    c.restoreState()
    
    c.saveState()
    c.translate(0, cursor)
    c.setFont("Helvetica", 20)
    c.drawCentredString(x, 0, dataString)
    cursor -= 1.3*cm    
    c.restoreState()
    
    cursor = SkipLine(cursor,1)
    
    # Sobre o Modelo
    cursor = InsertSection(c, 0, 'Modelo', cursor)
    
    table = [['Modelo:', modelName], 
              ['Número de Atributos:', len(colunas)]]
    cursor = InsertField(table, c, cursor)
    
    c.saveState()
    c.translate(7*cm, 12.6*cm)
    _ = InsertImage('images/feature_importance.png', c, 11*cm, 0)
    c.restoreState()
    
    cursor = SkipLine(cursor, n=3)
    c.showPage()
    
    ##########
    # Page 2 #
    ##########
    
    # About the Dataset    
    c.translate(MARGIN_X, 0)
    cursor = HEIGHT - MARGIN_Y
    cursor = InsertSection(c, 0, 'Dataset', cursor)
    
    # Unique Values on each column feature
    table2 = list()
    for col in uniqueValues.keys():
        table2.append([col, uniqueValues[col]])
    cursor = InsertField(table2, c, cursor)
    
    # Class Balance Plot
    NumToClass = lambda n: 'DP' if (n == 1) else 'DI'
    percentage = lambda n, total: (n * 100.0) / total
    
    total = sum(ClassBalance.values())    
    valueInPercentage = [percentage(x,total) for x in ClassBalance.values()]
    
    labels = list(map(NumToClass, ClassBalance.keys()))
    colors = ['#5B9BD5', '#BDD7EE']
              
    plt.pie(valueInPercentage, labels=labels, colors=colors, startangle=120,frame=False, autopct='%.1f %%')
    centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Balanceamento', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()          
    
    fig.savefig('images/class_balance_plot.png', bbox_inches='tight', transparent=True)
    
    # Show Class Balance Plot
    c.saveState()
    c.translate(8.5*cm, 18.3*cm)
    _ = InsertImage('images/class_balance_plot.png', c, 12*cm, 0)
    c.restoreState()
    
    # Codes distribution on the dataset
    c.saveState()
    c.translate(0*cm, MARGIN_Y)
    _ = InsertImage('images/code_frequency.png', c, 17.5*cm, 0)
    c.restoreState()
    
    c.showPage()
    
    ##########
    # Page 3 #
    ##########
    
    # Metrics - Test
    c.translate(MARGIN_X, 0)
    cursor = HEIGHT - MARGIN_Y
    sectionTitle = 'Métricas - Teste (%.2f %% - %d)' %(percentage(len(X_test),len(X)), len(X_test))
    cursor = InsertSection(c, 0, sectionTitle, cursor)
    
    # tabela com as métricas    
    table3 =[['Acurácia:', '%.2f %%' % (metrics_test['accuracy'] * 100)],
              ['MCC:', '%.5f' % (metrics_test['mcc'])],
              ['Macro-F1:', '%.5f' % (metrics_test['macrof1'])],
              ['Micro-F1:', '%.5f' % (metrics_test['microf1'])],
              ['AUC ROC:', '%.5f' % (metrics_test['rocauc'])]]
    cursor = InsertField(table3, c, cursor)
    
    
    c.saveState()
    c.translate(6*cm, 16*cm)
    _ = InsertImage('images/confusion_matrix_test.png', c, 12*cm, 0)
    c.restoreState()
    
    c.saveState()
    c.translate(0*cm, MARGIN_Y)
    _ = InsertImage('images/accuracy_by_code_test.png', c, 17.5*cm, 0)
    c.restoreState()
    
    c.showPage()
    
    ##########
    # Page 4 #
    ##########
    
    # Metrics - Train
    c.translate(MARGIN_X, 0)
    cursor = HEIGHT - MARGIN_Y
    sectionTitle = 'Métricas - Treino (%.2f %% - %d)' %(percentage(len(X_train),len(X)), len(X_train))
    cursor = InsertSection(c, 0, sectionTitle, cursor)
    
    table4 =[['Acurácia:', '%.2f %%' % (metrics_train['accuracy'] * 100)],
              ['MCC:', '%.5f' % (metrics_train['mcc'])],
              ['Macro-F1:', '%.5f' % (metrics_train['macrof1'])],
              ['Micro-F1:', '%.5f' % (metrics_train['microf1'])],
              ['AUC ROC:', '%.5f' % (metrics_train['rocauc'])]]
    cursor = InsertField(table4, c, cursor)
    
    c.saveState()
    c.translate(6*cm, 16*cm)
    _ = InsertImage('images/confusion_matrix_train.png', c, 12*cm, 0)
    c.restoreState()
    
    c.saveState()
    c.translate(0*cm, MARGIN_Y)
    _ = InsertImage('images/accuracy_by_code_train.png', c, 17.5*cm, 0)
    c.restoreState()
    
    c.showPage()
    c.save()