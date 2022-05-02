import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import sys
import os

from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.models import Sequential
import sys
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import datetime
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model

# Parámetros de entrada del script
region = sys.argv[1]
modelo = sys.argv[2]
window = sys.argv[3]

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

resultspath=fileDirectory+'\\results\\'+modelo +'\\'+region # Carpeta con las graficas de predicciones

def generar_train_test_datasets():
    dataURLRegiones='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/India_5_Regiones_Simultech3/'
    bmw_dataS=pd.read_excel('https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/india/total_bmw_waste.xlsx?raw=true',sheet_name=2)
    bmw_dataS['FECHA'] = pd.to_datetime(bmw_dataS['FECHA'], infer_datetime_format=True)
    bmw_dataS=bmw_dataS.fillna(bmw_dataS.mean())
    bmw_dataS=bmw_dataS.set_index('FECHA')

    if region == 'Puducherry':
        df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_Puducherry.csv')
    elif region == 'Goa':
        df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_Goa.csv')
    elif region == 'Manipur':
        df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_Manipur.csv')
    elif region == 'Nagaland':
        df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_Nagaland.csv')
    elif region == 'Mizoram':
        df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_Mizoram.csv')
    elif region == 'AMB':
        df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_AMB.csv')

    df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
    df_multivariable=df_multivariable.set_index('FECHA')
    df_real_data = df_multivariable.copy(deep=True)
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_data=X_scaler.fit_transform(df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']])
    Y_data = Y_scaler.fit_transform(df_multivariable[['BMW']])
    df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']]=X_data
    df_multivariable[['BMW']]=Y_data

    train_MRNN_sc=df_multivariable['2020-06-17':'2021-08-31']
    test_MRNN_sc=df_multivariable['2021-09-01':]
	
    test_MRNN_sc['BMW'][:]=0    #Unicamente para asegurar que las predicciones no esten tomando los valores reales de BMW 
    return train_MRNN_sc, test_MRNN_sc, bmw_dataS,  X_scaler, Y_scaler, df_real_data


def generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc):
    train_MRNN_scN=train_MRNN_sc.to_numpy()
    test_MRNN_scN=test_MRNN_sc.to_numpy()
    n_input = int(window)
    n_features = 6
    generator = TimeseriesGenerator(train_MRNN_scN, train_MRNN_scN, length=n_input, batch_size=1)   
    return generator,n_input,n_features,train_MRNN_scN, test_MRNN_scN

def generar_predicciones(model,train_MRNN_scN,test_MRNN_scN,Y_scaler,n_input,n_features):
    test_predictions = []
    test_variables=test_MRNN_scN
    temporal=[]

    first_eval_batch = train_MRNN_scN[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    test_batch= test_variables.reshape((1,len(test_variables),n_features))

    for i in range(len(test_MRNN_scN)):
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]
    
        # append the prediction into the array
        test_predictions.append(current_pred[n_input-1,0])

        test_batch[0,i,0]=current_pred[n_input-1,0]
        #test_batch[0,i,0]=current_pred
        temporal= test_batch[0,i,:]
        temporal= temporal.reshape((1,1,6))

        # use the prediction to update the batch and remove the first value
        #current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        current_batch = np.append(current_batch[:,1:,:],temporal,axis=1)
    
    true_predictions = Y_scaler.inverse_transform([test_predictions])
    Predicciones=[]

    for i in range(len(test_MRNN_scN)):
        Predicciones.append(true_predictions[0][i])

    return Predicciones

def plot_predicciones(testinverse,n_input,df_real_data):
    fig = plt.figure(figsize=(10, 6))

    plt.plot(df_real_data.index,df_real_data['BMW'], label="Real data")
    plt.plot(testinverse.index,testinverse['Predictions'],label="Predictions")
    plt.axvline(x=datetime.date(2021, 9, 1), ymin=-1, ymax=2,color="black",linestyle = "dashed",label="Start of the forecasting")

    plt.legend(loc='best')
    plt.title("Real predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for " + region)
    plt.xlabel("Date")
    plt.ylabel("BMW Tons")
    plt.savefig(resultspath+'\\predictions_real_'+region+'_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)
    


def plot_training(generator,model,Y_scaler,n_input,train_MRNN_sc,df_real_data):
    #evaluate for training 
    #train_data = windowed_dataset(train_escalado, window_size, batch_size=32, shuffle_buffer=0)
    trainRNNM_predict  = model.predict(generator)
    trainRNNM_predict=trainRNNM_predict[:,n_input-1,0].reshape(-1,1)

    trainRNNM_predict = Y_scaler.inverse_transform(trainRNNM_predict)
    trainRNNM_predict=pd.DataFrame(trainRNNM_predict ,index=train_MRNN_sc.index[n_input:],columns=['Test'])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df_real_data[n_input:-120].index,df_real_data['BMW'][n_input:-120], label="Real data")
    plt.plot(trainRNNM_predict.index,trainRNNM_predict['Test'],label="Predictions")
    plt.title("Train predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for " + region)
    ax.set_xlabel('Date')
    ax.set_ylabel("BMW Tons")
    ax.get_gid()
    ax.legend()
    plt.savefig(resultspath+'\\predictions_train_'+region+'_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)
    


def plot_test(test_MRNN_scN,model,Y_scaler,test_MRNN_sc,df_real_data):
    n_input = int(window)
    n_features = 6
    generatorTest = TimeseriesGenerator(test_MRNN_scN, test_MRNN_scN, length=n_input, batch_size=1)
    testRNNM_predict  = model.predict(generatorTest)
    testRNNM_predict=testRNNM_predict[:,n_input-1,0].reshape(-1,1)
    testRNNM_predict = Y_scaler.inverse_transform(testRNNM_predict)
    testRNNM_predict=pd.DataFrame(testRNNM_predict ,index=test_MRNN_sc.index[n_input:],columns=['Test'])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df_real_data[260:].index,df_real_data['BMW'][260:], label="Real data")
    plt.plot(testRNNM_predict.index,testRNNM_predict['Test'],label="Predictions")
    plt.title("Test predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for " + region)
    ax.set_xlabel('Date')
    ax.set_ylabel("BMW Tons")
    ax.get_gid()
    ax.legend()
    plt.savefig(resultspath+'\\predictions_test_'+region+'_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)

def plot_test2(testRNNM,model,Y_scaler,test_MRNN_sc,df_real_data):
    n_input = int(window)
    n_features = 6

    sc1 = MinMaxScaler(feature_range=(0,1))
    bmw_p = df_real_data['BMW']
    bmw_p = bmw_p.reshape(bmw_p.shape[0],1)
    bmw_p_scaled = sc1.fit_transform(bmw_p)
    testRNNM_predict  = model.predict(testRNNM)
    #testRNNM_predict=testRNNM_predict[:,n_input-1,0].reshape(-1,1)
    testRNNM_predict = sc1.inverse_transform(testRNNM_predict)
    #testRNNM_predict=pd.DataFrame(testRNNM_predict ,index=test_MRNN_sc.index[n_input:],columns=['Test'])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df_real_data[260:].index,df_real_data['BMW'][260:], label="Real data")
    plt.plot(testRNNM_predict.index,testRNNM_predict['Test'],label="Predictions")
    plt.title("Test predictions using the "+modelo + " model and a window-size of "+str(n_input) + " for " + region)
    ax.set_xlabel('Date')
    ax.set_ylabel("BMW Tons")
    ax.get_gid()
    ax.legend()
    plt.savefig(resultspath+'\\predictions_test2_'+region+'_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)
    
def windowed_dataset_multivariable(series, window_size, batch_size):
    # convert the series to tensor format
    dataset = tf.data.Dataset.from_tensor_slices(series)
    # create the windows shifted 1 position, 
    #drop_remainder keeps the number of elements in the window the same size.
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    # convert each window to numpy vector format
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # choose t --> t-1 as a features and the last one as a label
    # [0, 1, 2, 3][4]
    dataset = dataset.map(lambda window: (window[:-1], window[-1,0]))
    # shuffle the data
    #if (shuffle_buffer is not 0):
    #  print("shuffle")
    #  dataset = dataset.shuffle(shuffle_buffer)
    # Generate the batches
    # x = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # y = [[4], [8]]
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def main():
    train_MRNN_sc, test_MRNN_sc, bmw_dataS,  X_scaler, Y_scaler,df_real_data=generar_train_test_datasets()
    generator,n_input,n_features,train_MRNN_scN, test_MRNN_scN=generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc)
    #batch_size = 32
    region = sys.argv[1]
    modelo = sys.argv[2]

    if modelo == 'LSTM' or modelo== '':
        
        # define model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(256, return_sequences=True,activation='relu'))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(64, return_sequences=True,activation='relu'))
        #model.add(Dense(1,activation='linear'))
        model.add(LSTM(n_features, return_sequences=True)) 
        
    elif modelo == 'GRU':
        # define model
        model = Sequential()
        model.add(GRU(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
        model.add(GRU(128, return_sequences=True,activation='relu'))
        model.add(GRU(256, return_sequences=True,activation='relu'))
        model.add(GRU(128, return_sequences=True,activation='relu'))
        model.add(GRU(64, return_sequences=True,activation='relu'))
        #model.add(Dense(1,activation='linear'))
        model.add(GRU(n_features, return_sequences=True)) 
        
    elif modelo == 'RNN':
        # define model
        model = Sequential()
        model.add(SimpleRNN(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
        model.add(SimpleRNN(128, return_sequences=True,activation='relu'))
        model.add(SimpleRNN(256, return_sequences=True,activation='relu'))
        model.add(SimpleRNN(128, return_sequences=True,activation='relu'))
        model.add(SimpleRNN(64, return_sequences=True,activation='relu'))
        #model.add(Dense(16,activation='relu'))
        #model.add(Dense(1,activation='linear'))
        model.add(SimpleRNN(n_features, return_sequences=True)) 
      
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True
    model.compile(optimizer='adam', loss='mse')
    history=model.fit(generator,epochs=200)
    print("Model succesfully trained")
    
    mse = history.history['loss']

    epochs=range(len(mse)) # Get number of epochs
    #------------------------------------------------
    # Plot MAE, MSE and Loss
    #------------------------------------------------
    plt.figure(figsize=(10, 6));
    plt.plot(epochs, mse, 'green', label='MSE')
    plt.title("Number of epochs vs MSE using the "+modelo + " model and a window-size of "+str(n_input) + " for " + region)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()

    # Create subfolder for results
    isExist = os.path.exists(resultspath)

    if isExist != True:
        try:
            os.makedirs(resultspath)
            print(resultspath)
        except OSError:
            print("La creación del directorio  falló" )
        else:
            print("Se ha creado el directorio:  ")
       # os.makedirs(resultspath) # create results directory

    plt.savefig(resultspath+'\\epochs-mse_'+region+'_ws_'+ str(n_input)+"_"+modelo+"-model.png")
    
    Predicciones=generar_predicciones(model,train_MRNN_scN,test_MRNN_scN,Y_scaler,n_input,n_features)

    testinverse=pd.DataFrame(Y_scaler.inverse_transform(test_MRNN_sc), index=test_MRNN_sc.index,columns=test_MRNN_sc.columns)
    testinverse['Predictions']=Predicciones
    #testRNNM = windowed_dataset_multivariable(test_MRNN_sc, n_input, batch_size)
    plot_predicciones(testinverse,n_input,df_real_data)
    plot_training(generator,model,Y_scaler,n_input,train_MRNN_sc,df_real_data)
    #plot_test(test_MRNN_scN,model,Y_scaler,test_MRNN_sc,df_real_data)
    
    mse = mean_squared_error(df_real_data['BMW']['2021-09-01':'2021-12-31'], testinverse['Predictions'])
    errorMessage="Mean squared error using the "+modelo + " model and a window-size of "+str(window) + " for " + region + ': '+str(mse)
    with open("MSE.txt", "a") as f:
        print(errorMessage, file=f)
    
    #plot_test2(testRNNM,model,Y_scaler,test_MRNN_sc,df_real_data)
    #print(df_multivariable.head())
    return 0

if __name__ == '__main__':
    main()