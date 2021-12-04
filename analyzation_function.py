import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn
from scipy.signal import find_peaks
import math

def load_data(path="./data", start=1, end=250):
    """
    This function will load the file data in the specified range in order,
    the range is determined by the ID of the file
    All files are loaded by default
    
    Args:
        path: Data file storage location(relative path)
        start: File ID to start loading (default=1)
        end: File ID to end loading (default=250)
        
    Output:
        total_data, train_data, test_data, anomaly_start_data

    """
    path_list = os.listdir(path)
    path_list.sort()
    """Batch read and analyze the file data"""
    # you can change the this part to adjust the file range
    path_list=path_list[start-1:end]

    # load the file as data
    index = start #file index
    special_file_1, special_file_2 = set([204, 205]), set([206, 207, 208, 242, 243, 225, 226])#the index of different format data file 
    total_data = []
    train_data = []
    test_data = []
    anomaly_start_data = []
    for filename in path_list:
        temp1 = filename.split("_")
        #image_name =temp1[0] #get the index of file
        temp2 = temp1[-1].split(".")#get anomly start point
        anomaly_start_point = int(temp2[0])  
        #get Time Series data,split the train data and test data
        data = pd.read_csv(path+"/"+filename,names = ['values'])
        if index in special_file_1:
            df = pd.DataFrame()
            df['values'] = list(map(float, list(data.iloc[0])[0].strip(' ').split('   ')))
            data = df
        elif index in special_file_2:
            df = pd.DataFrame()
            df['values'] = list(map(float, list(data.iloc[0])[0].strip(' ').split('  ')))
            data = df
            
        total_data.append(data)
        train_data.append(data.iloc[:anomaly_start_point])
        test_data.append(data.iloc[anomaly_start_point:])
        anomaly_start_data.append(anomaly_start_point)
        index +=1
        #print("{}    data length: {} ".format(filename,data.size))
        #plot_time_series(data.values, anomaly_start_point,image_name)  #plot the image for each data
    return total_data,train_data,test_data,anomaly_start_data

def FindPeriod(arr):
    # the input should be a 1-D numpy array
    # initialize best period score
    best_score = float('inf')
    period = 0
    min_D = 60
    max_D = 300

    # D is the candidate period
    for d in range(min_D, max_D):

        # peaks locations
        p, _ = find_peaks(arr, distance=d)

        # valley locations
        v, _ = find_peaks(-arr, distance=d)

        # calculate interval lengths pd and vd from p and v
        pd = np.diff(p)
        vd = np.diff(v)

        # set current period d score
        cur_score = min(np.std(pd), np.std(vd)) / math.sqrt(d)

        if cur_score < best_score:
            best_score = cur_score
            period = d

    return period

def timeseries_z_normalize(data):
    """
    Normalize the time series data
    
    Output:
        the nomalized data(dataframe type)
    """
    data_mean = data.mean()
    data_std = data.std()
    data_normalize = (data - data_mean) / data_std
    return data_normalize

def timeseries_mean_normalize(data):
    """
    Normalize the time series data
    
    Output:
        the nomalized data(dataframe type)
    """
    data = np.array(data[0]['values'])
    data_mean = data.mean()
    data_min = data.min()
    data_max = data.max()
    data_normalize = (data - data_mean) / (data_max-data_min)
    
    return data_normalize

def create_sequences(data,time_steps):
    """
    Create sequence data
    Args:
        values (array): [time series data in array format ]
        time_steps (int): [time window for this data]. Defaults to 100.
    Output:
        the array
    """
    output = []
    file_num =1
    for i in range(file_num):
        for j in range( len(data)-time_steps):
            window_values = data[j:(j+time_steps)]
            output.append(window_values)
    return np.stack(output)

def plot_time_series(data, anomaly_start_point,image_name):
    #get plot parameter
    plt.figure(1,figsize=(200,15),dpi=100)

    x_1 = np.array(range(0,anomaly_start_point))
    x_2 = np.array(range(anomaly_start_point,len(data)))
    y_1 = data[0:anomaly_start_point].reshape(-1)
    y_2 = data[anomaly_start_point:len(data)].reshape(-1)
    
    xaxis_density = len(data)/100  #X-axis density factor

    #plot
    ax1=plt.subplot(111)
    plt.plot(x_1,y_1, color='blue')
    plt.plot(x_2,y_2, color='red')
    plt.xlim(0,len(data))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_density))#set x axis density
    plt.xticks(rotation=60)

    plt.savefig('./image/'+image_name+'.jpg') #save path,you can change the name
    plt.show()
    
def plot_simple_sequence(model_input, model_output,saved_name,epoch):
    model_input=model_input.reshape(-1,1).cpu().detach().numpy()
    model_output=model_output.reshape(-1,1).cpu().detach().numpy()

    plt.title('EPOCH '+str(epoch) + ' result')

    plt.plot(np.arange(len(model_input)), model_input, label = 'origin',color='grey')
    plt.plot(np.arange(len(model_output)), model_output, label = 'reconstruct', color='blue')
    plt.plot(np.arange(len(model_input)), abs(model_input - model_output)-1, label = 'anomaly score',color='red')
    plt.legend()
    plt.savefig('./window_recon_everyepoch/'+saved_name+'.jpg')
    #plt.show()
    plt.clf()
    
def reconstruct_AE(length, window_size, inputs, model,device) -> np.ndarray:
    i = 0
    reconstruct_X = np.zeros(shape=length)
    cnt = np.zeros(shape=length)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
    while i < (len(inputs)):
        #randomly sample some images' latent vectors from its distribution
        pred = model(inputs[i]).detach().cpu().numpy()
        reconstruct_X[i: i + window_size] += pred
        cnt[i: i + window_size] += 1
        i += window_size // 10 #reconstruct every xx windows
    cnt_zero = [int(x == 0) for x in cnt] 
    cnt_final = cnt_zero+cnt
    print("Reconstruct done")
    return reconstruct_X / cnt_final 

def reconstruct_ConvAE(length, window_size, inputs, model,device) -> np.ndarray:
    i = 0
    reconstruct_X = np.zeros(shape=length)
    cnt = np.zeros(shape=length)
    inputs = torch.from_numpy(inputs.astype(np.float32)).to(device)
       
    while i < (len(inputs)):
        #randomly sample some images' latent vectors from its distribution
        pred_inputs = inputs[i].reshape(1,1,len(inputs[i])) 
        pred = model(pred_inputs).detach().cpu().numpy()
        pred = pred.reshape(window_size,)
        reconstruct_X[i: i + window_size] += pred
        cnt[i: i + window_size] += 1
        i += window_size // 10 #reconstruct every xx windows
    cnt_zero = [int(x == 0) for x in cnt] 
    cnt_final = cnt_zero+cnt
    print("Reconstruct done")
    return reconstruct_X / cnt_final 