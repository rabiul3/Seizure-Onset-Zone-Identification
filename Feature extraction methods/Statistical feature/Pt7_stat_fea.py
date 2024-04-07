# Average Amplitude Change
from sklearn import preprocessing

def coeffofvariation(X,parr):
 
    X=X*parr
    L = len(X)
    
    miu = (1/L)*sum(np.abs(X))  # calculating mean
    
    sd = sqrt((1/L)*sum(np.power((X-(1/L)*sum(X)),2))) # calculating standard deviation
 
      
    cv = pow(sd,2)/pow(miu,2)
    
    return cv
 
   # result1 = pow(x, y) #python power
    
    #C = power(A,B)
def fluctuationindex(X,parr):
    
    X=X*parr
    fi = sum(np.abs(np.diff(X)))/len(X);
    return fi



def jMAV(X,parr):
    X=X*parr
    MAV=np.mean(abs(X))   ### Absoulate mean
    return MAV




def jRMS(X,parr):
    X=X*parr
    RMS1=(np.power(X, 2))   ### Root mean square signal
    RMS2=(np.mean(RMS1))
    
    RMS=sqrt(RMS2)
    return RMS



#Difference Absolute Standard Deviation Value
def jDASDV(X,parr):
    X=X*parr

    N=len(X)
    Y=0
    for i in range(N-1):
        Y=Y+(X[i+1]-X[i])**2
    
    DASDV=sqrt(Y/[N-1]) 
    return DASDV





#Log Detector
def jLD(X,parr):
    X=X*parr
    
    N=len(X)
    Y=0
    for k in range(N):
        Y=Y+log(abs(X[k]))
    
    LD=exp(Y/N)
    return LD



#Modified Mean Absolute Value
def jMMAV(X,parr):
    X=X*parr
    
    N=len(X)
    Y=0
    for i in range(N):
        if i >= 0.25*N and i <= 0.75*N:
            w=1        
        else:
            w=0.5
        Y=Y+(w*abs(X[i]))
    MAV1=(1/N)*Y
    return MAV1



# Modified Mean Absolute Value 2
def jMMAV2(X,parr):
    X=X*parr
    
    N=len(X)
    Y=0
    for i in range(N):
        if (i >= 0.25*N) and (i <= 0.75*N):
            w=1
        elif i < 0.25*N:
            w=(4*i)/N
        else:
            w=4*(i-N)/N
        
        Y=Y+(w*abs(X[i]))
    MMAV2=(1/N)*Y
    return MMAV2





# Variance 
def jVAR(X,parr):
    
    X=X*parr
    
    N=len(X)
    VAR=(1/(N-1))*sum(np.power(X,2))
    return VAR

 
    

    
    
    
#butter worth band pass filter
from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a



def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b,a, data)
    return y


# butter bandpass filter call for all bands

def filter_bank(data,fs):    
    B1 = butter_bandpass_filter(data, 100, 150, fs, order=3)

    B2 = butter_bandpass_filter(data, 150, 200, fs, order=3)
    
    B3 = butter_bandpass_filter(data, 200, 250, fs, order=3)
    
    B4 = butter_bandpass_filter(data, 250, 300, fs, order=3)
    
    B5 = butter_bandpass_filter(data, 300, 350, fs, order=3)
    
    B6 = butter_bandpass_filter(data, 350, 400, fs, order=3)
    
    B7 = butter_bandpass_filter(data, 400, 450, fs, order=3)
    
    B8 = butter_bandpass_filter(data, 450, 500, fs, order=3)
    
    B9 = butter_bandpass_filter(data, 500, 550, fs, order=3)
    
    B10 = butter_bandpass_filter(data, 550, 600, fs, order=3)
    
    bank_filter=np.vstack((B1,B2,B3,B4,B5,B6,B7,B8,B9,B10))
    
    return bank_filter





#All new three feature 

def new_sir_feature(data):
    #thres=0.016
    #parr=10e5 
    parr=1
    cv=coeffofvariation(data,parr)
    
    fi=fluctuationindex(data,parr)
    
    MAV=jMAV(data,parr)
    
    RMS=jRMS(data,parr)

    #AAC = jAAC(data,parr)
    DASDV = jDASDV(data,parr)
    LD = jLD(data,parr)
    MAV1 = jMMAV(data,parr)
    MMAV2 = jMMAV2(data,parr)
    #MYOP = jMYOP(data,parr)
    VAR = jVAR(data,parr)
    
    new_fea = np.hstack((cv,fi, MAV, RMS,DASDV, LD, MAV1, MMAV2, VAR ))
    return new_fea

import numpy as np
import scipy as sp
import scipy.io 
import os
import mne
from mne import io
#import pan

# EDF file read from mne


EDF_PATH = '/Volumes/ExternalHDD/intracranial/FCD/00_out_201811_EEG_10993749/'   # This is the path of data of data in Juntendo server
EDF_NAME = EDF_PATH +'DJ0010M5.edf'


data_FCD2 = io.read_raw_edf(EDF_NAME, preload=True, stim_channel=None)
#b=len(data_FCD2)
#b=len(data_FCD2)


#To  cut the signal from 30min s to 2 hours




a=71 #number of channel 42
s=0  # starting time 30 mmin 1s 
f=40000 # dividing into 20s
Temp_test = np.zeros(shape=[a,40000])  #creating an matrix and set all value zero or two dimensional array  size=aX40000 where a=20
#data_FCD2_n=np.zeros(shape=[a,14400078])   #creating an matrix and set all value zero  size=aX14400077  where a=20

data_FCD2_n,time=data_FCD2[range(0,71)]




################################ Data Normalization #######################

data_FCD2_n = preprocessing.normalize(data_FCD2_n)


##########################################################################



#1 hour 1s to end 
Frame1_test=np.zeros((180,a,40000))  #creating an matrix or three dimensional array and setting all value zero  size=360XaX40000 where a=20



# This line is only for only patients 3

#chan0_36=data_FCD2_n[0:37,:]
#chan39_44=data_FCD2_n[39:44,:]

#data_FCD2_n=np.vstack((chan0_36, chan39_44))

        
for sg in range(180): # for test set we used 30 mins


    Temp_test=data_FCD2_n[:,s:f] 

    Frame1_test[sg,:,:]=Temp_test  

    s+=40000
    f+=40000

#Reshape Frame1 from 3 dimension to 2 dimension   
    
#new_Frame_test = Frame1_test.reshape((Frame1_test.shape[0]*Frame1_test.shape[1]), Frame1_test.shape[2]) 

#print("b")



# all segments and all channel filtered bank

from datetime import datetime
from math import *
import numpy as np
import scipy as sp
import scipy.io 
import cmath

from scipy.fftpack import fft, ifft

fs=2000


segmented_frame=np.empty(shape=[71,40000])
stored_filtered_data=np.empty(shape=[71,10,40000])
seg_filtered_data=np.empty(shape=[180,71,10,40000])

for i in range(180): #number of segments
    segmented_frame=Frame1_test[i,:,:]
    for j in range(71):
        
        stored_filtered_data[j,:,:]=filter_bank(segmented_frame[j,:],fs)
        
    seg_filtered_data[i,:,:,:]=stored_filtered_data
        
        
    

 #seg_filtered_data=contain filterbank result   

#Apply new three feature on each subbands

Temp1= np.empty(shape=[10,9])       #subbans X features
Temp2=np.empty(shape=[71,10,9])     #channel X subbands X new_features
Temp3=np.empty(shape=[180,71,10,9])  #segments X cnannel X subbands X new_features

thres=0.016

for filtered_seg in range(180):
    q=np.squeeze(seg_filtered_data[filtered_seg,:,:,:])  
    print("Segment_{0}".format(filtered_seg))
    
    for ch in range(71):
        filtered_ch=np.squeeze(q[ch,:,:])
        
        
        for sub_band in range(10):
            
            Temp1[sub_band,:]=new_sir_feature(filtered_ch[sub_band,:])
            #print( Temp1.shape)
            
        Temp2[ch,:,:]=Temp1
        
    Temp3[filtered_seg,:,:,:]=Temp2
        
  
        
scipy.io.savemat('P7_sheuli_new_fea_0_180epoc_norm.mat', mdict={'p7_0_180epoc_sheuliFea_fea': Temp3})        
    