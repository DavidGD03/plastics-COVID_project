from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import random
import time

import matplotlib.patches as patches
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import sys
from sklearn import datasets


from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import sys
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import datetime
from sklearn.metrics import mean_squared_error



def generar_train_test_datasets():
    dataURL='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/'
    bmw_dataS=pd.read_excel(dataURL+'india/total_bmw_waste.xlsx?raw=true',sheet_name=2)
    bmw_dataS['FECHA'] = pd.to_datetime(bmw_dataS['FECHA'], infer_datetime_format=True)
    bmw_dataS=bmw_dataS.fillna(bmw_dataS.mean())
    bmw_dataS=bmw_dataS.set_index('FECHA')


    if sys.argv[1] == 'Puducherry':
        df_multivariable=pd.read_csv(dataURL+'India_5_Regiones_Simultech2/df_multivariable_Puducherry.csv')
        df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
        df_multivariable=df_multivariable.set_index('FECHA')
    elif sys.argv[1] == 'Goa':
        df_multivariable=pd.read_csv(dataURL+'India_5_Regiones_Simultech2/df_multivariable_Goa.csv')
        df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
        df_multivariable=df_multivariable.set_index('FECHA')
    elif sys.argv[1] == 'Manipur':
        df_multivariable=pd.read_csv(dataURL+'India_5_Regiones_Simultech2/df_multivariable_Manipur.csv')
        df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
        df_multivariable=df_multivariable.set_index('FECHA')
    elif sys.argv[1] == 'Nagaland':
        df_multivariable=pd.read_csv(dataURL+'India_5_Regiones_Simultech2/df_multivariable_Nagaland.csv')
        df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
        df_multivariable=df_multivariable.set_index('FECHA')
    elif sys.argv[1] == 'Mizoram':
        df_multivariable=pd.read_csv(dataURL+'India_5_Regiones_Simultech2/df_multivariable_Mizoram.csv')
        df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
        df_multivariable=df_multivariable.set_index('FECHA')
    elif sys.argv[1] == 'AMB':
        df_multivariable=pd.read_csv(dataURL+'India_5_Regiones_Simultech2/df_multivariable_AMB.csv')
        df_multivariable['FECHA'] = pd.to_datetime(df_multivariable['FECHA'], infer_datetime_format=True)
        df_multivariable=df_multivariable.set_index('FECHA')

    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_data=X_scaler.fit_transform(df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']])
    Y_data = Y_scaler.fit_transform(df_multivariable[['BMW']])
    df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']]=X_data
    df_multivariable[['BMW']]=Y_data

    train_MRNN_sc=df_multivariable['2020-06-17':'2021-02-01']
    test_MRNN_sc=df_multivariable['2021-02-02':]

    test_MRNN_sc['BMW'][:]=0    #Unicamente para asegurar que las predicciones no esten tomando los valores reales de BMW 

    #print(sys.argv[1])
    #print(df_multivariable.head())
    return train_MRNN_sc, test_MRNN_sc, bmw_dataS,  X_scaler, Y_scaler, df_multivariable


def generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc):
    train_MRNN_scN=train_MRNN_sc.to_numpy()
    test_MRNN_scN=test_MRNN_sc.to_numpy()
    n_input = int(sys.argv[3])
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

        #print("temporal:")
        #print(temporal)
        # use the prediction to update the batch and remove the first value
        #current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        current_batch = np.append(current_batch[:,1:,:],temporal,axis=1)
        #print(current_batch)
    
    true_predictions = Y_scaler.inverse_transform([test_predictions])
    Predicciones=[]

    for i in range(0,148):
        Predicciones.append(true_predictions[0][i])

    return Predicciones

def plot_predicciones(bmw_dataS,testinverse,n_input,df_multivariable):
    fig = plt.figure(figsize=(10, 6))

    plt.plot(df_multivariable.index,df_multivariable['BMW'], label="Real data")
    plt.plot(testinverse.index,testinverse['Predictions'],label="Predictions")


    plt.axvline(x=datetime.date(2021, 2, 1), ymin=-1, ymax=2,color="black",linestyle = "dashed",label="Start of the forecasting")

    plt.legend(loc='best')
    plt.title("Real predictions using the "+sys.argv[2] + " model and a window-size of "+sys.argv[1])
    plt.xlabel("Date")
    plt.ylabel("BMW Tons")
    plt.savefig('predictions_real_'+sys.argv[1]+'_ws_'+ str(n_input)+"_"+sys.argv[2]+"-model.png",dpi=fig.dpi)
    plt.show()
    


def plot_training(bmw_dataS,generator,model,Y_scaler,n_input,train_MRNN_sc,df_multivariable):
    #evaluate for training 
    #train_data = windowed_dataset(train_escalado, window_size, batch_size=32, shuffle_buffer=0)

    trainRNNM_predict  = model.predict(generator)
    trainRNNM_predict=trainRNNM_predict[:,n_input-1,0].reshape(-1,1)

    trainRNNM_predict = Y_scaler.inverse_transform(trainRNNM_predict)
    trainRNNM_predict=pd.DataFrame(trainRNNM_predict ,index=train_MRNN_sc.index[n_input:],columns=['Test'])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df_multivariable[n_input:-150].index,df_multivariable['BMW'][n_input:-150], label="Real data")
    plt.plot(trainRNNM_predict.index,trainRNNM_predict['Test'],label="Predictions")
    plt.title("Train predictions using the "+sys.argv[2] + " model and a window-size of "+sys.argv[1])
    ax.set_xlabel('Date')
    ax.set_ylabel("BMW Tons")
    ax.get_gid()
    ax.legend()
    plt.savefig('predictions_train_'+sys.argv[1]+'_ws_'+ str(n_input)+"_"+sys.argv[2]+"-model.png",dpi=fig.dpi)
    plt.show()
    


def plot_test(test_MRNN_scN,model,Y_scaler,test_MRNN_sc,bmw_dataS):
    n_input = int(sys.argv[3])
    n_features = 6
    generatorTest = TimeseriesGenerator(test_MRNN_scN, test_MRNN_scN, length=n_input, batch_size=1)
    testRNNM_predict  = model.predict(generatorTest)
    testRNNM_predict=testRNNM_predict[:,n_input-1,0].reshape(-1,1)
    testRNNM_predict = Y_scaler.inverse_transform(testRNNM_predict)
    testRNNM_predict=pd.DataFrame(testRNNM_predict ,index=test_MRNN_sc.index[n_input:],columns=['Test'])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df_multivariable[230:].index,df_multivariable['BMW'][230:], label="Real data")
    plt.plot(testRNNM_predict.index,testRNNM_predict['Test'],label="Predictions")
    plt.title("Test predictions using the "+sys.argv[2] + " model and a window-size of "+sys.argv[1])
    ax.set_xlabel('Date')
    ax.set_ylabel("BMW Tons")
    ax.get_gid()
    ax.legend()
    plt.savefig('predictions_test_'+sys.argv[1]+'_ws_'+ str(n_input)+"_"+sys.argv[2]+"-model.png",dpi=fig.dpi)
    plt.show()
    


def main():
    train_MRNN_sc, test_MRNN_sc, bmw_dataS,  X_scaler, Y_scaler,df_multivariable=generar_train_test_datasets()
    generator,n_input,n_features,train_MRNN_scN, test_MRNN_scN=generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc)


    if sys.argv[2] == 'LSTM' or sys.argv[2]== '':
        
        # define model
        model = Sequential()
        model.add(LSTM(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(256, return_sequences=True,activation='relu'))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(64, return_sequences=False,activation='relu'))
        model.add(Dense(1),activation='linear')
        #model.add(LSTM(n_features, return_sequences=True)) 
        
    elif sys.argv[2] == 'GRU':
        # define model
        model = Sequential()
        model.add(GRU(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
        model.add(GRU(128, return_sequences=True,activation='relu'))
        model.add(GRU(256, return_sequences=True,activation='relu'))
        model.add(GRU(128, return_sequences=True,activation='relu'))
        model.add(GRU(64, return_sequences=False,activation='relu'))
        model.add(Dense(1),activation='linear')
        #model.add(GRU(n_features, return_sequences=True)) 
        
    elif sys.argv[2] == 'RNN':
        # define model
        model = Sequential()
        model.add(SimpleRNN(64, return_sequences=True,activation='tanh', input_shape=(n_input, n_features)))
        model.add(SimpleRNN(128, return_sequences=True,activation='tanh'))
        model.add(SimpleRNN(256, return_sequences=True,activation='tanh'))
        model.add(SimpleRNN(128, return_sequences=True,activation='tanh'))
        model.add(SimpleRNN(64, return_sequences=False,activation='tanh'))
        model.add(Dense(16),activation='relu')
        model.add(Dense(1),activation='linear')
        #model.add(SimpleRNN(n_features, return_sequences=True)) 
      

    model.compile(optimizer='adam', loss='mse')
    model.fit(generator,epochs=50)
    print("Model succesfully trained")
    Predicciones=generar_predicciones(model,train_MRNN_scN,test_MRNN_scN,Y_scaler,n_input,n_features)

    testinverse=pd.DataFrame(Y_scaler.inverse_transform(test_MRNN_sc), index=test_MRNN_sc.index,columns=test_MRNN_sc.columns)
    testinverse['Predictions']=Predicciones

    plot_predicciones(bmw_dataS,testinverse,n_input,df_multivariable)
    plot_training(bmw_dataS,generator,model,Y_scaler,n_input,train_MRNN_sc,df_multivariable)
    plot_test(test_MRNN_scN,model,Y_scaler,test_MRNN_sc,bmw_dataS,df_multivariable)
    mse = mean_squared_error(df_multivariable['BMW']['2021-02-02':'2021-06-29'], testinverse['Predictions'])
    print("Mean squared error: ",mse)


    #print(df_multivariable.head())
    return 0

if __name__ == '__main__':
    main()
