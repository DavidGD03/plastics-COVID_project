import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.preprocessing import MinMaxScaler
import datetime

# Parámetros de entrada del script
region = 'AMB'
modelo = 'LSTM'
window = '7'

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

resultspath=fileDirectory+'\\results\\'+modelo +'\\'+region # Carpeta con las graficas de predicciones

def generar_train_test_datasets(escenario,X_scaler,Y_scaler):
    dataURLRegiones='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/India_5_Regiones_Simultech3/'
    # dataURLSEIRD='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/SEIRD_AMB_Paper_Old/'
    # dataURLSEIRD='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/SEIRD_AMB_Paper/'
    dataURLSEIRD='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/SEIRD_AMBv4/'
    bmw_dataS=pd.read_excel('https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/india/total_bmw_waste.xlsx?raw=true',sheet_name=2)
    bmw_dataS['FECHA'] = pd.to_datetime(bmw_dataS['FECHA'], infer_datetime_format=True)
    bmw_dataS=bmw_dataS.fillna(bmw_dataS.mean())
    bmw_dataS=bmw_dataS.set_index('FECHA')
    df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_AMB.csv')
    df_multivariable=df_multivariable.set_index('FECHA')
    if escenario == 'pessimistic':
        df_IPesimista=pd.read_csv(dataURLSEIRD+'ISEIRDPESIMISTA.csv',names=['FECHA','AMB'],header=0, index_col='FECHA')
        df_DPesimista=pd.read_csv(dataURLSEIRD+'DSEIRDPESIMISTA.csv',names=['FECHA','AMB'],header=0, index_col='FECHA')
        df_multivariable['Casos']=df_IPesimista['AMB']
        df_multivariable['Muertes']=df_DPesimista['AMB']
    elif escenario == 'neutral':
        df_INeutral=pd.read_csv(dataURLSEIRD+'ISEIRDNEUTRAL.csv',names=['FECHA','AMB'],header=0, index_col='FECHA')
        df_DNeutral=pd.read_csv(dataURLSEIRD+'DSEIRDNEUTRAL.csv',names=['FECHA','AMB'],header=0, index_col='FECHA')
        df_multivariable['Casos']=df_INeutral['AMB']
        df_multivariable['Muertes']=df_DNeutral['AMB']
    elif escenario == 'optimistic':
        df_IOptimista=pd.read_csv(dataURLSEIRD+'ISEIRDOPTIMISTA.csv',names=['FECHA','AMB'],header=0, index_col='FECHA')
        df_DOptimista=pd.read_csv(dataURLSEIRD+'DSEIRDOPTIMISTA.csv',names=['FECHA','AMB'],header=0, index_col='FECHA')
        df_multivariable['Casos']=df_IOptimista['AMB']
        df_multivariable['Muertes']=df_DOptimista['AMB']
    df_multivariable.index = pd.to_datetime(df_multivariable.index, infer_datetime_format=True)
    X_data=X_scaler.transform(df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']])
    Y_data = Y_scaler.transform(df_multivariable[['BMW']])
    df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']]=X_data
    df_multivariable[['BMW']]=Y_data
    train_MRNN_sc=df_multivariable['2020-06-17':'2021-08-31']
    test_MRNN_sc=df_multivariable['2021-09-01':]
    test_MRNN_sc['BMW'] = 0    #Unicamente para asegurar que las predicciones no esten tomando los valores reales de BMW 
    return train_MRNN_sc, test_MRNN_sc, Y_scaler

def generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc):
    train_MRNN_scN=train_MRNN_sc.to_numpy()
    test_MRNN_scN=test_MRNN_sc.to_numpy()
    n_input = int(window)
    n_features = 6
    return n_input, n_features, train_MRNN_scN, test_MRNN_scN

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
        temporal= test_batch[0,i,:]
        temporal= temporal.reshape((1,1,6))
        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:,1:,:],temporal,axis=1)
    true_predictions = Y_scaler.inverse_transform([test_predictions])
    Predicciones=[]
    for i in range(len(test_MRNN_scN)):
        Predicciones.append(true_predictions[0][i])
    return Predicciones

def plot_predicciones(testinverse, n_input, lista_escenarios, df_real_data):
    fig = plt.figure(figsize=(10, 6))
    for escenario in lista_escenarios:
        plt.plot(testinverse.index,testinverse['predictions_' + escenario],label="Predictions for the " + escenario + " scenario")
    plt.axvline(x=datetime.date(2021, 9, 1), ymin=-1, ymax=2,color="black",linestyle = "dashed",label="Start of the forecasting")
    plt.plot(df_real_data.index,df_real_data['BMW'], label="Real data")
    plt.legend(loc='best')
    plt.title("Real predictions using the "+modelo + " model and a window-size of "+str(n_input)+ " for " + region)
    plt.xlabel("Date")
    plt.ylabel("BMW Tons")
    plt.savefig(resultspath+'\\predictions_real_'+region+'_ws_'+ str(n_input)+"_"+modelo+"-scenarios.png",dpi=fig.dpi)

def main():
    dataURLRegiones='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/India_5_Regiones_Simultech3/'
    bmw_dataS=pd.read_excel('https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/india/total_bmw_waste.xlsx?raw=true',sheet_name=2)
    bmw_dataS['FECHA'] = pd.to_datetime(bmw_dataS['FECHA'], infer_datetime_format=True)
    bmw_dataS=bmw_dataS.fillna(bmw_dataS.mean())
    bmw_dataS=bmw_dataS.set_index('FECHA')
    df_multivariable=pd.read_csv(dataURLRegiones+'df_multivariable_AMB.csv')
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_scaler=X_scaler.fit(df_multivariable[['Casos', 'Muertes', 'Mov Residencial', 'Mov trabajo', 'Mov estaciones']])
    Y_scaler=Y_scaler.fit(df_multivariable[['BMW']])
    lista_escenarios = ['pessimistic', 'neutral', 'optimistic']
    train_MRNN_sc, test_MRNN_sc, Y_scaler=generar_train_test_datasets('pessimistic',X_scaler,Y_scaler)
    testinverse=pd.DataFrame(Y_scaler.inverse_transform(test_MRNN_sc), index=test_MRNN_sc.index,columns=test_MRNN_sc.columns)
    for escenario in lista_escenarios:
        train_MRNN_sc, test_MRNN_sc, Y_scaler=generar_train_test_datasets(escenario,X_scaler,Y_scaler)
        n_input,n_features,train_MRNN_scN, test_MRNN_scN=generador_serie_tiempo(train_MRNN_sc,test_MRNN_sc)
        model = tf.keras.models.load_model(fileDirectory + '/LSTM_model.h5')
        Predicciones=generar_predicciones(model,train_MRNN_scN,test_MRNN_scN,Y_scaler,n_input,n_features)
        testinverse['predictions_' + escenario]=Predicciones

    # Create subfolder for results
    isExist = os.path.exists(resultspath)
    if isExist != True:
        try:
            os.makedirs(resultspath)
            print(resultspath)
        except OSError:
            print("La creación del directorio  falló" )
        else:
            print("Se ha creado el directorio correctamente ")

    dataURLRegiones='https://raw.githubusercontent.com/DavidGD03/plastics-COVID_project/main/data/India_5_Regiones_Simultech3/'
    df_real_data=pd.read_csv(dataURLRegiones+'df_multivariable_AMB.csv')
    df_real_data['FECHA'] = pd.to_datetime(df_real_data['FECHA'], infer_datetime_format=True)
    df_real_data=df_real_data.set_index('FECHA')
    plot_predicciones(testinverse,n_input, lista_escenarios, df_real_data)
    return 0

if __name__ == '__main__':
    main()