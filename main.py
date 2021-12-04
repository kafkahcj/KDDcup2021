import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from analyzation_function import *
from VAE import *
from AE import *
from CNN_AE import *
from CNN_VAE import *
import torch
import torch.nn.functional as F
import argparse
import shutil
import os
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#torch.set_default_tensor_type(torch.FloatTensor)

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--result_dir', type=str, default='./model_result',
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./model_save',
					help='model saving directory')
parser.add_argument('--resume', default='./model_save', type=str,
					help='path to latest checkpoint (default: None')

parser.add_argument('--loss_show_every', default=200, type=int,
					help='show loss every steps(windows)')
parser.add_argument('--num_worker', type=int, default=0,
					help='num_worker')

# model options
parser.add_argument('--batch_size', type=int, default=128,
					help='input batch size for training (default: 128)')
parser.add_argument('--epoch_num', type=int, default=50,
					help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--z_dim', type=int, default=32,
					help='latent vector size of encoder')
parser.add_argument('--input_dim', type=int, default=150,
					help='input dimension, the time window of the time sequence')
parser.add_argument('--input_channel', type=int, default=1,
					help='input channel, 1 for each data')

args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists('./test_residual'):
    os.makedirs('./test_residual')
if not os.path.exists('./window_recon_everyepoch'):
    os.makedirs('./window_recon_everyepoch')


def save_checkpoint(state,is_best,outdir,saved_index):
    
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	checkpoint_file = os.path.join(outdir, 'checkpoint_'+str(saved_index)+'.pth')
	best_file = os.path.join(outdir, 'model_best_'+str(saved_index)+'.pth')
	torch.save(state, checkpoint_file)
	if is_best:
            shutil.copyfile(checkpoint_file, best_file)

def train_vae(train_dataloader,test_dataloader,input_dim, z_dim, learning_rate):
    writer = SummaryWriter(log_dir='./VAE_logs', comment='training_writer')
    global file_index #record the file index
    #set model
    model = VAE(input_dim=input_dim, z_dim=z_dim).to(device)
    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #early_stopping = EarlyStopping(patience=50, verbose=True)
    
    best_test_loss = np.finfo('f').max
    #train model
    loss_list=[]
    for epoch in range(args.epoch_num):
        train_avg_loss = 0
        for i,data in enumerate(train_dataloader):
            model.train()
            inputs = data[0].to(device).float()               
            #forward
            res, mu, logvar = model(inputs)
            
            ## get loss(VAE)
            loss,recon_loss,KLD = loss_function_vae(res,inputs,mu,logvar)
            train_avg_loss += loss
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print training result
            if (i+1) % args.loss_show_every ==0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.8f} Total loss {:.8f}"
                    .format(epoch + 1, args.epoch_num, i + 1, len(train_dataloader), recon_loss.item(),
                            KLD.item(), loss.item()))
            #plot the image
            if i == 0:
                saved_name ='EPOCH '+ str(epoch)
                plot_simple_sequence(inputs,res,saved_name,epoch)
        
        ###  test  ###
        test_avg_loss =0.0
        with torch.no_grad():
            for idx, test_data in enumerate(test_dataloader):
                #get inputs
                test_inputs = test_data[0].to(device).float()    
                #forward
                test_res,test_mu,test_logvar = model(test_inputs)
                test_loss,test_recon_loss,test_KLD = loss_function_vae(res,inputs,mu,logvar)
                test_avg_loss += test_loss
                
        
        #plot the train/test loss
        train_loss_epoch = train_avg_loss/len(train_dataloader.dataset)
        test_loss_epoch = test_avg_loss/len(test_dataloader.dataset)
        writer.add_scalars('VAE_loss_{}'.format(file_index),{"train_loss":train_loss_epoch,
                                        "test_loss":test_loss_epoch},epoch)
        #print the epoch loss
        print("Epoch[{}/{}],train loss:{}, test loss:{}".format(epoch+1,args.epoch_num,train_loss_epoch,test_loss_epoch))

                
def train_AE(train_dataloader,test_dataloader,input_dim, z_dim, learning_rate):
    writer = SummaryWriter(log_dir='./AE_logs', comment='training_writer')
    global file_index #record the file index
    start_epoch =0
    #set model
    model = AE(input_dim=input_dim, z_dim=z_dim).to(device)
    best_model=False
    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #early_stopping = EarlyStopping(patience=50, verbose=True)
    best_test_loss = np.finfo('f').max
    
	# optionally resume from a checkpoint
    checkpoint_file = args.resume+'/checkpoint_'+str(file_index)+'.pth'
    if args.resume:
        if os.path.isfile(checkpoint_file):
            print('=> loading checkpoint %s' % checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % checkpoint_file)
            best_model = model
        else:
            print('=> no checkpoint found at %s' % args.resume)
            
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    ###  train model  ###
    for epoch in range(start_epoch, args.epoch_num):
        train_avg_loss = 0
        for i,data in enumerate(train_dataloader):
            model.train()
            inputs = data[0].to(device).float()               
            #forward
            res = model(inputs)
            
            ## get loss(AE)
            loss = F.mse_loss(res, inputs)
            train_avg_loss += loss
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print the step loss
            if (i+1) % args.loss_show_every ==0:
                print("Epoch[{}/{}], Step [{}/{}],loss {:.8f}"
                    .format(epoch + 1, epoch_num, i + 1, len(train_dataloader), loss.item()))
            #plot the image
            if i == 0:
                saved_name ='EPOCH '+ str(epoch)
                plot_simple_sequence(inputs,res,saved_name,epoch)
        

        ###  test  ###
        test_avg_loss =0.0
        with torch.no_grad():
            for idx, test_data in enumerate(test_dataloader):
                #get inputs
                test_inputs = test_data[0].to(device).float()    
                #forward
                test_res = model(test_inputs)
                test_loss = F.mse_loss(test_res, test_inputs)
                test_avg_loss += test_loss
        
        #plot the train/test loss
        train_loss_epoch = train_avg_loss/len(train_dataloader.dataset)
        test_loss_epoch = test_avg_loss/len(test_dataloader.dataset)
        writer.add_scalars('final_loss_AE_funcwindow'+str(file_index),{"train_loss":train_loss_epoch,
                                         "test_loss":test_loss_epoch},epoch)
        #print the epoch loss
        print("Epoch[{}/{}],train loss:{}, test loss:{}".format(epoch+1,epoch_num,train_loss_epoch,test_loss_epoch))
                                         

        #save model
        
        if test_loss_epoch < best_test_loss:
            is_best =True
            best_model = model
            best_test_loss = test_loss_epoch
            
            save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            }, is_best, args.save_dir,file_index)
            
        
    return best_model

def train_ConvAE(train_dataloader,test_dataloader,input_dim, z_dim, learning_rate):
    writer = SummaryWriter(log_dir='./ConvAE_logs', comment='training_writer')
    global file_index #record the file index
    start_epoch =0
    #set model
    model = ConvAE(input_channel=1, input_dim=input_dim,z_dim=z_dim).to(device)
    best_model=False
    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #early_stopping = EarlyStopping(patience=50, verbose=True)
    best_test_loss = np.finfo('f').max
    
	# optionally resume from a checkpoint
    """"""
    checkpoint_file = args.resume+'/checkpoint_'+str(file_index)+'.pth'
    if args.resume:
        if os.path.isfile(checkpoint_file):
            print('=> loading checkpoint %s' % checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % checkpoint_file)
            best_model = model
        else:
            print('=> no checkpoint found at %s' % args.resume)
            

    
    ###  train model  ###
    for epoch in range(start_epoch, args.epoch_num):
        train_avg_loss = 0
        for i,data in enumerate(train_dataloader):
            model.train()
            inputs = data[0].to(device).float()  
            inputs = inputs.reshape(1,1,input_dim)
            #forward
            res = model(inputs)
            
            ## get loss(AE)
            loss = F.mse_loss(res, inputs)
            train_avg_loss += loss
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print the step loss
            if (i+1) % args.loss_show_every ==0:
                print("Epoch[{}/{}], Step [{}/{}],loss {:.8f}"
                    .format(epoch + 1, epoch_num, i + 1, len(train_dataloader), loss.item()))
            #plot the image
            if i == 0:
                saved_name ='EPOCH '+ str(epoch)
                plot_simple_sequence(inputs,res,saved_name,epoch)
        

        ###  test  ###
        test_avg_loss =0.0
        with torch.no_grad():
            for idx, test_data in enumerate(test_dataloader):
                #get inputs
                test_inputs = test_data[0].to(device).float()
                test_inputs = test_inputs.reshape(1,1,input_dim)    
                #forward
                test_res = model(test_inputs)
                test_loss = F.mse_loss(test_res, test_inputs)
                test_avg_loss += test_loss
        
        #plot the train/test loss
        train_loss_epoch = train_avg_loss/len(train_dataloader.dataset)
        test_loss_epoch = test_avg_loss/len(test_dataloader.dataset)
        writer.add_scalars('final_train_loss_ConvAE_'+str(file_index),{"train_loss":train_loss_epoch,
                                         "test_loss":test_loss_epoch},epoch)
        #print the epoch loss
        print("Epoch[{}/{}],train loss:{:.8f}, test loss:{:.8f}".format(epoch+1,epoch_num,train_loss_epoch,test_loss_epoch))
                                         

        #save model
        if test_loss_epoch < best_test_loss:
            is_best =True
            best_model = model
            best_test_loss = test_loss_epoch
            
            save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            }, is_best, args.save_dir,file_index)
            
        
    return best_model

def train_ConvVAE(train_dataloader,test_dataloader,input_dim, z_dim, learning_rate):
    writer = SummaryWriter(log_dir='./ConvVAE_logs', comment='training_writer')
    #set model
    model = ConvVAE(input_channel=1, input_dim=input_dim,z_dim=z_dim).to(device)
    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #early_stopping = EarlyStopping(patience=50, verbose=True)
    
    best_test_loss = np.finfo('f').max
    #train model
    loss_list=[]
    for epoch in range(args.epoch_num):
        train_avg_loss = 0
        for i,data in enumerate(train_dataloader):
            model.train()
            inputs = data[0].to(device).float()     
            inputs = inputs.reshape(1,1,input_dim)   
                   
            #forward
            res, mu, logvar = model(inputs)
            
            ## get loss(VAE)
            loss,recon_loss,KLD = loss_function_vae(res,inputs,mu,logvar)
            train_avg_loss += loss
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print training result
            if (i+1) % args.loss_show_every ==0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.8f} Total loss {:.8f}"
                    .format(epoch + 1, args.epoch_num, i + 1, len(train_dataloader), recon_loss.item(),
                            KLD.item(), loss.item()))
            #plot the image
            if i == 0:
                saved_name ='EPOCH '+ str(epoch)
                plot_simple_sequence(inputs,res,saved_name,epoch)
        
        ###  test  ###
        test_avg_loss =0.0
        with torch.no_grad():
            for idx, test_data in enumerate(test_dataloader):
                #get inputs
                test_inputs = test_data[0].to(device).float()    
                test_inputs = test_inputs.reshape(1,1,input_dim)
                #forward
                test_res,test_mu,test_logvar = model(test_inputs)
                test_loss,test_recon_loss,test_KLD = loss_function_vae(res,inputs,mu,logvar)
                test_avg_loss += test_loss
                
        
        #plot the train/test loss
        train_loss_epoch = train_avg_loss/len(train_dataloader.dataset)
        test_loss_epoch = test_avg_loss/len(test_dataloader.dataset)
        writer.add_scalars('ConvVAE_loss',{"train_loss":train_loss_epoch,
                                        "test_loss":test_loss_epoch},epoch)
        #print the epoch loss
        print("Epoch[{}/{}],train loss:{}, test loss:{}".format(epoch+1,args.epoch_num,train_loss_epoch,test_loss_epoch))
        

if __name__ == '__main__':
    
    #load data
    val_file=[1,5,11,13,14,16,20,22,23,29,30,43,56,72,99,127,128,132,133,238]
    val_file_anomaly_location=[52100,5390,11900,16030,17000,17000,7200,6600,8640,4540,4200,19360,11180,52700,3674,6200,7320,4560,5626,41661]
    k=0
    
    for i in range(1,251):   #train and reconstruct on all file
    #for i in val_file:  #train and reconstruct on  validation file
        #get file index
        file_index = i
        #print file information
        print('-----NewRound-----\nProcessing file '+str(file_index))
        #get validation anomaly location
        val_anomaly_location = val_file_anomaly_location[k]
        k+=1
        
        #get the data
        total_data,train_data,test_data,anomaly_start_data =\
        load_data(path="../data-sets/KDD-Cup/data", start=file_index,end=file_index)
        
        #set parameter
        epoch_num = args.epoch_num
        learning_rate = args.lr
        z_dim = args.z_dim
        input_dim = args.input_dim
        
        #normalization the data 
        total_data_norm = timeseries_mean_normalize(total_data)
        data_length = len(total_data_norm)
        print('file length: ',data_length)
        #### get window size ###
        """You can choose use fixed window size or use fucntion-window size here"""
        ##set fixed window ##
        window_size = args.input_dim
        
        ##function window ##
        #window_size = FindPeriod(total_data_norm)
        #input_dim =window_size
        
        print('file'+str(i)+': window size='+str(window_size))
        
        #Extract data series by time window
        train_data = total_data_norm[:anomaly_start_data[0]]
        test_data = total_data_norm[anomaly_start_data[0]:]
        train_sequence = create_sequences(train_data,window_size)
        test_sequence = create_sequences(test_data,window_size)

        #get dataloader
        train_dataloader = DataLoader(dataset=train_sequence, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
        
        ### train model ###
        """You can choose one model here. PS:the VAE/ConvVAE cannot output the final reconstructed results,
            but you could see the reconstruction result in a window in the file folder
        """
        #train_vae(train_dataloader,train_dataloader,input_dim,z_dim,learning_rate) #vae
        #train_ConvVAE(train_dataloader,train_dataloader,input_dim,z_dim,learning_rate=0.0001)  #cnn+vae
        
        #model = train_AE(train_dataloader,train_dataloader,input_dim,z_dim,learning_rate) #ae
        model = train_ConvAE(train_dataloader,train_dataloader,input_dim,z_dim,learning_rate)  #cnn+ae
        
        
        ### Reconstruct ###
        total_sequence = create_sequences(total_data_norm,window_size)
        #reconstruct function for AE
        #reconstruct_X = reconstruct_AE(data_length, window_size, total_sequence, model,device) 
        #reconstruct function for ConvAE
        reconstruct_X = reconstruct_ConvAE(data_length, window_size, total_sequence, model,device) 
        
        #get the reconstruction 
        train_y = reconstruct_X[:anomaly_start_data[0]]
        test_y = reconstruct_X[anomaly_start_data[0]:]
        
        #compute the timeseries residual
        times_residual = np.abs(total_data_norm-reconstruct_X)
        times_residual[-50:]=0 #Avoid end outliers affecting results
        times_residual_testpart = times_residual[anomaly_start_data[0]:]
        
        #get the anomaly point location
        anomaly_point_index = np.argmax(times_residual_testpart)+anomaly_start_data[0]
        print('Validation anomaly location: ',val_anomaly_location)
        print('Predicted anomaly location: ',anomaly_point_index)
        anomaly_point_value = total_data_norm[anomaly_point_index]
        
        #compute the distance from the validation location
        dis_index_to_true_anomaly = np.abs(anomaly_point_index-val_anomaly_location)                
        if dis_index_to_true_anomaly <=100:
            model_score = 100
        else:
            model_score = dis_index_to_true_anomaly
            
        print('File '+str(i)+"model score: "+str(model_score))
        
        #write down the test timeseries residual
        file = open('./test_residual/file_'+str(i)+'_'+str(model_score)+'.txt','w')
        for j in times_residual_testpart:
            file.write(str(j)+'\n')
        file.close()
        
        #plot the reconstruction
        plt.figure(figsize=(7,5),dpi=300)
        ax1=plt.subplot(111)
        plt.title(str(file_index) + 'th file')
        plt.plot(np.arange(len(total_data_norm)), total_data_norm, label = 'origin')
        plt.plot(np.arange(len(train_y)), train_y, label = 'reconstruct_train')
        plt.plot(np.arange(len(train_y),len(total_data_norm)), test_y, label = 'reconstruct_test')
        plt.plot(np.arange(len(reconstruct_X)), times_residual - 1, color='grey',label = 'anomaly score')
        plt.axvline(x=anomaly_point_index,color='red', linestyle=':',linewidth=1,label='Predicted anomaly') #plot the predicted anomaly point line
        #plot the true anomaly (for the validation), if run the model on all files, just comment out here
        plt.axvline(x=val_anomaly_location,color='blue', linestyle=':',linewidth=1,label='True anomaly') #plot the predicted anomaly point line

        #set xaxis 
        ax1.set_xlim(0,len(total_data_norm))
        xaxis_density = len(total_data_norm)/25  #X-axis density factor
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_density))#set x axis density
        plt.xticks(rotation='90')
        #set yaxis
        y_major_locator = MultipleLocator(0.1) # The Y-axis scale is a multiple of 0.1 
        ax1.yaxis.set_major_locator(y_major_locator)
        #set legend location
        #ax1.legend(loc='center left', bbox_to_anchor=(1.001, 0.5),ncol=1)
        #plt.legend()
        plt.savefig('./model_result/'+'recon_file_'+str(file_index)+'_score_'+str(model_score)+'.jpg',dpi=300)
        #plt.show()
        plt.clf()
       
        