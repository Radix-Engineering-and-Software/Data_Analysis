import numpy as np
import pandas as pd
from Methods import util
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
import pickle
import warnings


# Functions for data preprocessing 
        
def LabelEncode(df, columns = None, save = False, load = False, path = '', returnEncoders = False):
    """
    Performs label encoding on specified columns or in all of them.
    
    - df (pandas dataframe/ pandas Series): dataframe containing the columns to 
    label encode, or series with only one column.
    - columns (list of strings/numbers) : list of columns names to label encode
    - save (bool): tells if the encoder must be saved.
    - load (bool): tells if the encoder must be loaded.
    - path (string): path where the encoder must be saved at or loaded from.
    - returnEncoders (bool): tells if the encoder should be returned too.
    
    Return (pandas dataframe, list of LabelEncoders): dataset with specified 
    columns label encoded and the Label Encoder itself in case returnEncoder is 
    True.
    """
    if isinstance(df,pd.Series):
        df = df.to_frame()
        
    if columns is None:
        columns = list(df.columns)  

    # if directory doesnt exist, create it
    util.CheckAndCreatePath(path)
    
    encoders = list()
    for column in columns:
        # Load or create Label Encoder
        if load:
            encoder = pickle.load(open(path + "label_encoder_" + 
                                      str(column) + ".pickle.dat", "rb"))
        else:
            encoder = preprocessing.LabelEncoder()
        
        df[column] = encoder.fit_transform(df[column])            
        encoders.append(encoder)
        
        # Save encoder
        if save:
            pickle.dump(encoder, open(path + "label_encoder_" + 
                                      str(column) + ".pickle.dat", "wb")) 
            
    if returnEncoders:
        return df, encoders
    
    return df

def OneHotEncode(df, columns = None, save = False, load = False, path = ''):
    """
    Performs One-Hot encoding on specified columns or in all of them.
    
    - df (pandas dataframe): dataframe containing the columns to one-hot encode
    - columns (list of strings/numbers) : list of columns names to one-hot encode
    - save (bool): bool that tells if the encoder must be saved.
    - load (bool): bool that  tells if the encoder must be loaded.
    - path (string): path where the encoder must be saved at or loaded from.
    
    Return (pandas dataframe): dataframe with specified columns label encoded.
    """
    if isinstance(df,pd.Series):
        df = df.to_frame()
        
    if columns is None:
        columns = list(df.columns) 
        
    # if directory doesnt exist, create it
    util.CheckAndCreatePath(path)   
        
    # Perform Label encode on the columns
    df_enc, labelEncoders = LabelEncode(df, columns = columns, save = save, load = load, path = path, returnEncoders=True)
    
    # Perform One-hot enconde on the columns
    cont = 0            
    for column in columns:
        if load:
            encoder = pickle.load(open(path + "one_hot_encoder_" + 
                                      str(column) + ".pickle.dat", "rb"))
        else:
            encoder = preprocessing.OneHotEncoder()
                    
        oneHotEncoded = encoder.fit_transform(df_enc[column].to_frame())
        
        # Create datafame with the one-hot encoded features
        columns_transformed = [str(column)+'_'+str(i) for i in labelEncoders[cont].classes_]
        oneHotEncoded = pd.DataFrame(oneHotEncoded.toarray(), columns=columns_transformed)
        
        # Put one-hot columns on the right position of the dataset and delete old feaure
        df_enc = util.InsertColumnsOnADataframePosition(df_enc, oneHotEncoded, list(df_enc.columns).index(column))
        df_enc = df_enc.drop(columns=[column])
        
        # Save encoder
        if save:
            pickle.dump(encoder, open(path + "one_hot_encoder_" + 
                                      str(column) + ".pickle.dat", "wb"))             
        cont += 1
        
    return df_enc


def ColumnAsBagOfWords(column, regex = None, save = False, load = False, 
                       applyTFIDF = True, path = '', **kwargs):
    """
    Encode text column as bag of words.
    
    - colum (pd.Series): vector containing the text data.
    - regex (raw string): string with the tokenizer pattern. If not passed, a
    pattern where everything but "_", "." and " " is accepted as word is assumed
    - save (bool): bool that tells if the Vectorizer must be saved.
    - load (bool): bool that  tells if the Vectorizer must be loaded.
    - applyTDIDF (bool): bool that  tells if td-idf (term frequency–inverse document frequency)
    should be applied
    - path (string): path where the Vectorizer must be saved at or loaded from.
    - kwags(dict): dictionary of CountVectorizer/TfidfTransformer parameters to set.
    
    Return (pd.Dataframe, CountVectorizer, TfidfTransformer): Resulting dataframe,
    Count Vectorizer (scikit-learn), Tfidf Transformer (scikit-learn)
    """
    # if directory doesnt exist, create it
    util.CheckAndCreatePath(path)  
        
    if regex is None:
        regex = r"[^_^.^-]+"
    
    ColumnName = column.name
    
    # Apply bag of words vectorization
    if load:
        count_vect = pickle.load(open(path + "count_vectorizer_" + 
                                      str(ColumnName) + ".pickle.dat", "rb"))
    else:
        count_vect = CountVectorizer(token_pattern = regex)
    
    # Capture count vectorizer parameters from kwags
    try:
        count_vect.set_params(**kwargs)
    except:
        pass
        
    columTransformed = count_vect.fit_transform(column)
    
    # Apply tf-idf(term frequency–inverse document frequency)
    if applyTFIDF:
        if load:
            tf_idf = pickle.load(open(path + "tf_idf_" + 
                                      str(ColumnName) + ".pickle.dat", "rb"))
        else:
            tf_idf = TfidfTransformer(norm='l1', use_idf=True)
    
        # Capture Tfidf Transformer parameters from kwags
        try:
            tf_idf.set_params(**kwargs)
        except:
            pass
    
        columTransformed = tf_idf.fit_transform(columTransformed)
    else:
        tf_idf = None

    # create vector of column names    
    columnNames = [(str(ColumnName) + '-' + str(i)) for i in count_vect.vocabulary_.keys()]
    
    # Save vectorizer and tf-idf transformer
    if save:
        pickle.dump(count_vect, open(path + "count_vectorizer_" + 
                                  str(ColumnName) + ".pickle.dat", "wb"))
        if applyTFIDF:
            pickle.dump(tf_idf, open(path + "tf_idf_" + str(ColumnName) + 
                                     ".pickle.dat", "wb"))
    
    # Construct final dataframe of transformed column
    df = pd.DataFrame(columTransformed.toarray(), columns=columnNames)
    
    return df, count_vect, tf_idf

def ApplyPCA(X, columns, save=False, load=False, path = '', n_components = 9):
    '''
    Apply PCA on specific columns on the dataset
    
    - X (pd.DataFrame): dataset with the columns where PCA should be applied.
    - columns (list of numbers): List of number Ids of columns to apply the PCA.
    - save (bool): bool that tells if the PCA transoformer should be saved.
    - load (bool): bool that tells if the PCA transoformer should be loaded.
    - path (string): path where the Vectorizer must be saved at or loaded from.
    - n_components (int): number of PCA components.
    
    Return X_train(pd.DataFrame): X_train with the transformed features.
    '''
    
    # if directory doesnt exist, create it
    util.CheckAndCreatePath(path)            
    
    # Apply PCA
    if load:
        pca = pickle.load(open(path + "pca_transformer.pickle.dat", "rb"))
    else:
        pca = PCA(n_components=n_components)
    
    selectedColumns = X.columns[columns]    
    pcaFeatures = pca.fit_transform(X[selectedColumns])
    columnsNames = list(range(pcaFeatures.shape[1]))
    pcaFeatures = pd.DataFrame(pcaFeatures, columns=columnsNames)    
        
#    X_transformed = pd.concat([pcaFeatures, X.drop(selectedColumns, axis=1)], axis=1)
    X_transformed = util.ConcatenateDataframes(pcaFeatures, X.drop(selectedColumns, axis=1))
        
    # Save PCA transformer
    if save:
        pickle.dump(pca, open(path + "pca_transformer.pickle.dat", "wb"))
        
    return X_transformed
    
def SeparateAddress(df, target='Procedência'):
    """
    Extract City and Neighbourhood fields from the address, and construct features.
    
    - df (pd.DataFrame): Entire dataset including target feature.
    - target (string): Name of the target feature.
    
    Return (pd.DataFrame): Dataset with the new columns if sucessfull.
    """
    # Clean Address field (remove pontuctiation, slashes and double spaces)    
    df['ECF_STREET_ADDRESS'] = df['ECF_STREET_ADDRESS'].str.replace(r'\s*[,./]\s*', ' ')
    df['ECF_STREET_ADDRESS'] = df['ECF_STREET_ADDRESS'].str.replace(r'\s{2,}', ' ')
    df['ECF_STREET_ADDRESS'] = df['ECF_STREET_ADDRESS'].str.replace(r'\s+$', '')
    
    # Get City and Neighbourhood of the address
    city = []
    neighbourhood = []    
    try:
        for i in range(0,len(df.loc[:,'ECF_STREET_ADDRESS'])):    
            words = df.loc[i,'ECF_STREET_ADDRESS'].split(":")
            city.append(words[-2])
            neighbourhood.append(words[-1])
        
        df['cidade'] = pd.Series(city)
        df['bairro'] = pd.Series(neighbourhood)
        
        # Return columns in the right order (target is the last one)
        columns = [i for i in df.columns if i != target]
        columns.append(target)
        return df[columns]
    except:
        warnings.warn("Address columns were not properly treated.")
        return df

def splitInTrainAndTestSet(X, Y, train_perc = 0.7):
    """
    Split Data into Train and test sets.
    
    - X (pandas.DataFrame): Input Dataframe.
    - Y (pandas.Series): Labels Dataframe.
    - train_perc (float): number between 0 and 1 that repesents the ratio of 
    the dataset which should be used as the train set.
    
    Return (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series): X_train, Y_train,
    X_Test, Y_test
    """
    size_of_data = len(Y) 
    train_ini = 0
    train_end = int(train_perc*size_of_data)
    test_ini = train_end
    test_end = size_of_data

    X_train = X.iloc[train_ini:train_end,:]
    Y_train = Y[train_ini:train_end]

    X_test = X.iloc[test_ini:test_end,:]
    Y_test = Y[test_ini:test_end]
    
    return X_train, Y_train, X_test, Y_test

def RemoveNaNsAndWhiteSpaces(df):
    """
    Subsitutes NaNs and white spaces on the dataset by '0'.
    
    - df (pd.DataFrame): Dataframe that represents the dataset
    
    Return (pd.DataFrame)
    """
    df = df.replace(r'^\s+$', np.nan, regex=True)
    df = df.fillna('0')
    return df
