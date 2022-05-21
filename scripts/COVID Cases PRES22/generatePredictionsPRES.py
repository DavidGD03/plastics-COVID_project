import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
from datetime import datetime
import sys
import os

# Parámetros de entrada del script
periodo = sys.argv[1] # Si es 1 la predicción empieza en julio 2021, si es 2 empieza en octubre 2021
modelo = sys.argv[2]
window = sys.argv[3]

# Parámetros modelo
media=True # Aplicar filtro media móvil

# Fechas de predicción
if periodo=='1' or periodo=='':
    fechaITrain='2020-05-01'
    fechaFTrain='2021-06-30'
    fechaITest='2021-07-01'
    fechaFTest='2021-09-30'
elif periodo=='2':
    fechaITrain='2020-05-01'
    fechaFTrain='2021-09-30'
    fechaITest='2021-10-01'
    fechaFTest='2021-12-31'


absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

resultspath=fileDirectory+'/results/'+modelo # Carpeta con las graficas de predicciones

def preproceDatosAMB(ciudad):
    df=pd.read_csv("https://www.datos.gov.co/resource/gt2j-8ykr.csv?$limit=1000000&ciudad_municipio_nom="+ciudad)
    data = df.copy()
    data["fecha_diagnostico"] = data["fecha_diagnostico"].replace({'7/10/2020 0:00:01':'7/10/2020 0:00:00'}) # Corrección de incosistencia en el dataset
    data["fecha_diagnostico"] = pd.to_datetime(data["fecha_diagnostico"], infer_datetime_format=True)
    contagiados = data[data["fecha_diagnostico"].isna()!=True]
    contagiados = contagiados["fecha_diagnostico"].groupby(contagiados["fecha_diagnostico"]).count()
    contagiados = contagiados.sort_index()
    #contagiados = contagiados/pob_ciudades[pob_ciudades["Ciudad"]==ciudad]['Población'].values[0]
    contagiadosdf=pd.DataFrame(contagiados, columns= ['fecha_diagnostico'])
    contagiadosdf.index = pd.to_datetime(contagiadosdf.index, infer_datetime_format=True)
    return contagiadosdf

def generar_train_test_datasets(media, fechaITrain, fechaFTrain, fechaITest, fechaFTest):
    contagiadosBmanga=preproceDatosAMB('BUCARAMANGA')
    contagiadosFlorida=preproceDatosAMB('FLORIDABLANCA')
    contagiadosGiron=preproceDatosAMB('GIRON')
    contagiadosPiedecuesta=preproceDatosAMB('PIEDECUESTA')

    contagiadosAMB=contagiadosBmanga.copy()
    contagiadosAMB= contagiadosAMB.add(contagiadosFlorida, fill_value=0)
    contagiadosAMB=contagiadosAMB.add(contagiadosGiron, fill_value=0)
    contagiadosAMB=contagiadosAMB.add(contagiadosPiedecuesta, fill_value=0)

    data = contagiadosAMB.copy()

    # Media móvil
    if media:
            data=data.rolling(window=7).mean()
            data=data.dropna()

    #division del dataset en train y test
    df = pd.DataFrame(data)
    df.rename(columns = {'fecha_diagnostico':'cases'}, inplace = True)
    df_real_data = df.copy()
    df_train=df[fechaITrain:fechaFTrain]
    df_test=df[fechaITest:fechaFTest]

    X_scaler = MinMaxScaler(feature_range=(0,1))
    Y_scaler = MinMaxScaler(feature_range=(0,1))
    X_data=X_scaler.fit_transform(df_train)
    Y_data=Y_scaler.fit_transform(df_test)
    train_MRNN_sc=pd.DataFrame(X_data,index=df_train.index,columns=['cases'])
    test_MRNN_sc=pd.DataFrame(Y_data,index=df_test.index,columns=['cases'])

    test_MRNN_sc['cases'][:]=0    #Unicamente para asegurar que las predicciones no esten tomando los valores reales de BMW 

    return train_MRNN_sc, test_MRNN_sc, X_scaler, Y_scaler, df_real_data


def generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc):
    train_MRNN_scN=train_MRNN_sc.to_numpy()
    test_MRNN_scN=test_MRNN_sc.to_numpy()
    n_input = int(window)
    n_features = 1
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
        temporal= temporal.reshape((1,1,1))
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
    df_real_data=df_real_data[fechaITest:fechaFTest]
    plt.plot(df_real_data.index,df_real_data['cases'], label="Real data")
    plt.plot(testinverse.index,testinverse['Predictions'],label="Predictions")

    if periodo=='1' or periodo=='':
        plt.axvline(x=datetime.date(2021, 7, 1), ymin=-1, ymax=2,color="black",linestyle = "dashed",label="Start of the forecasting")
    elif periodo=='2':
        plt.axvline(x=datetime.date(2021, 10, 1), ymin=-1, ymax=2,color="black",linestyle = "dashed",label="Start of the forecasting")

    plt.legend(loc='best')
    plt.title("Real predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for the AMB")
    plt.xlabel("Date")
    plt.ylabel("COVID-19 Cases")
    plt.savefig(resultspath+'/predictions_real_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)

def plot_training(generator,model,Y_scaler,n_input,train_MRNN_sc,df_real_data):
    trainRNNM_predict  = model.predict(generator)
    trainRNNM_predict=trainRNNM_predict[:,n_input-1,0].reshape(-1,1)
    trainRNNM_predict = Y_scaler.inverse_transform(trainRNNM_predict)
    trainRNNM_predict=pd.DataFrame(trainRNNM_predict ,index=train_MRNN_sc.index[n_input:],columns=['Test'])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(df_real_data[n_input:-120].index,df_real_data['cases'][n_input:-120], label="Real data")
    plt.plot(trainRNNM_predict.index,trainRNNM_predict['Test'],label="Predictions")
    plt.title("Train predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for the AMB")
    ax.set_xlabel('Date')
    plt.ylabel("COVID-19 Cases")
    ax.get_gid()
    ax.legend()
    plt.savefig(resultspath+'/predictions_train_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)

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
    plt.title("Test predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for the AMB")
    ax.set_xlabel('Date')
    ax.set_ylabel("BMW Tons")
    ax.get_gid()
    ax.legend()
    plt.savefig(resultspath+'/predictions_test_ws_'+ str(n_input)+"_"+modelo+"-model.png",dpi=fig.dpi)

def main():
    epochs=150
    train_MRNN_sc, test_MRNN_sc, X_scaler, Y_scaler,df_real_data=generar_train_test_datasets(media, fechaITrain, fechaFTrain, fechaITest, fechaFTest)
    generator,n_input,n_features,train_MRNN_scN, test_MRNN_scN=generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc)
    modelo = sys.argv[2]
    capas = sys.argv[4] # Si es 1 se usan muchas capas, si es 2 se usan menos capas en las redes neuronales
    if modelo == 'LSTM' or modelo== '':
        model = Sequential()
        if capas=='1':
            model.add(LSTM(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
            model.add(LSTM(128, return_sequences=True,activation='relu'))
            model.add(LSTM(256, return_sequences=True,activation='relu'))
            model.add(LSTM(128, return_sequences=True,activation='relu'))
            model.add(LSTM(64, return_sequences=True,activation='relu'))
        elif capas=='2':
            model.add(LSTM(32, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
            model.add(LSTM(32, return_sequences=True,activation='relu'))
        model.add(LSTM(n_features, return_sequences=True)) 
        
    elif modelo == 'GRU':
        model = Sequential()
        if capas=='1':
            model.add(GRU(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
            model.add(GRU(128, return_sequences=True,activation='relu'))
            model.add(GRU(256, return_sequences=True,activation='relu'))
            model.add(GRU(128, return_sequences=True,activation='relu'))
            model.add(GRU(64, return_sequences=True,activation='relu'))
        elif capas=='2':
            model.add(GRU(32, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
            model.add(GRU(32, return_sequences=True,activation='relu'))
        model.add(GRU(n_features, return_sequences=True)) 
        
    elif modelo == 'RNN':
        model = Sequential()
        if capas=='1':
            model.add(SimpleRNN(64, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
            model.add(SimpleRNN(128, return_sequences=True,activation='relu'))
            model.add(SimpleRNN(256, return_sequences=True,activation='relu'))
            model.add(SimpleRNN(128, return_sequences=True,activation='relu'))
            model.add(SimpleRNN(64, return_sequences=True,activation='relu'))
        elif capas=='2':
            model.add(SimpleRNN(32, return_sequences=True,activation='relu', input_shape=(n_input, n_features)))
            model.add(SimpleRNN(32, return_sequences=True,activation='relu'))
        model.add(SimpleRNN(n_features, return_sequences=True)) 

    model.compile(optimizer='adam', loss='mse')
    history=model.fit(generator,epochs=epochs)
    print("Model succesfully trained")
    
    mse = history.history['loss']
    epochs=range(len(mse)) # Get number of epochs
    #------------------------------------------------
    # Plot MAE, MSE and Loss
    #------------------------------------------------
    plt.figure(figsize=(10, 6));
    plt.plot(epochs, mse, 'green', label='MSE')
    plt.title("Number of epochs vs MSE using the "+modelo + " model and a window-size of "+str(n_input) + " for the AMB")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()

    # Create subfolder for results
    isExist = os.path.exists(resultspath)

    if isExist != True:
        try:
            os.makedirs(resultspath)
        except OSError:
            print("La creación del directorio falló" )
        else:
            print("Se ha creado el directorio:  "+resultspath)

    plt.savefig(resultspath+'/epochs-mse_ws_'+ str(n_input)+"_"+modelo+"-model.png")
    model.save(resultspath+'/model_ws_'+ str(n_input)+"_"+modelo+".h5")

    Predicciones=generar_predicciones(model,train_MRNN_scN,test_MRNN_scN,Y_scaler,n_input,n_features)
    testinverse=pd.DataFrame(Y_scaler.inverse_transform(test_MRNN_sc), index=test_MRNN_sc.index,columns=test_MRNN_sc.columns)
    testinverse['Predictions']=Predicciones
    plot_predicciones(testinverse,n_input,df_real_data)
    plot_training(generator,model,Y_scaler,n_input,train_MRNN_sc,df_real_data)
    #plot_test(test_MRNN_scN,model,Y_scaler,test_MRNN_sc,df_real_data)
    
    mse = mean_squared_error(df_real_data['cases'][fechaITest:fechaFTest], testinverse['Predictions'])
    errorMessage="Mean squared error using the "+modelo + " model and a window-size of "+str(window) + " for the AMB: "+str(mse)
    with open("MSE.txt", "a") as f:
        print(errorMessage, file=f)
    return 0

if __name__ == '__main__':
    main()