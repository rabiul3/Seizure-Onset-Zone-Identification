from sklearn import preprocessing
def coeffofvariation(sig,parr):
 
    sig=sig*parr
    L = len(sig)
    
    miu = (1/L)*sum(np.abs(sig))  # calculating mean
    
    sd = sqrt((1/L)*sum(np.power((sig-(1/L)*sum(sig)),2))) # calculating standard deviation
 
      
    cv = pow(sd,2)/pow(miu,2)
    
    return cv
 
   # result1 = pow(x, y) #python power
    
    #C = power(A,B)     #matlab power
    
    
    

def fluctuationindex(sig,parr):
    
    sig=sig*parr
    fi = sum(np.abs(np.diff(sig)))/len(sig);
    return fi



def jMAV(sig,parr):
    sig=sig*parr
    MAV=np.mean(abs(sig))   ### Absoulate mean
    return MAV




def jRMS(sig,parr):
    sig=sig*parr
    RMS1=(np.power(sig, 2))   ### Root mean square signal
    RMS2=(np.mean(RMS1))
    
    RMS=sqrt(RMS2)
    return RMS


# Average Amplitude Change
def jAAC(X,parr):
    
    X=X*parr
    N=len(X)
    Y=0
    M0=X[0]

    for i in range(N-1):
        Y=Y+abs(X[i+1]-X[i])

    AAC=Y/N    
    return AAC




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



# Myopulse Percentage Rate
def jMYOP(X,parr):
    
    X=X*parr
    N=len(X)
    Y=0
    for i in range(N):
        if abs(X[i]) >= thres:
            Y=Y+1
    MYOP=Y/N
    return MYOP



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
    
    #B8 = butter_bandpass_filter(data, 450, 500, fs, order=3)
    
    #B9 = butter_bandpass_filter(data, 500, 550, fs, order=3)
    
    #B10 = butter_bandpass_filter(data, 550, 600, fs, order=3)
    
    bank_filter=np.vstack((B1,B2,B3,B4,B5,B6,B7))
    
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


EDF_NAME= '/Volumes/ExternalHDD/intracranial/FCD/20200511_tanaka/EDF/FJ8130O9.edf'

#channel_array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58])


data_FCD2 = io.read_raw_edf(EDF_NAME, preload=True, stim_channel=None)
#b=len(data_FCD2)
#b=len(data_FCD2)


#To  cut the signal from 30min s to 2 hours




a=60 #number of channel 42
s=0  # starting time 30 mmin 1s 
f=20000 # dividing into 20s
Temp_test = np.zeros(shape=[a,20000])  #creating an matrix and set all value zero or two dimensional array  size=aX40000 where a=20
#data_FCD2_n=np.zeros(shape=[a,14400078])   #creating an matrix and set all value zero  size=aX14400077  where a=20

data_FCD2_n,time=data_FCD2[range(0,67)]



#data_FCD2_n=data_FCD2_n[:,7200000:10800000]

#number of channel 60 and  start time from 
#1 hour 1s to end 
Frame1_test=np.zeros((180,a,20000))  #creating an matrix or three dimensional array and setting all value zero  size=360XaX40000 where a=20


chan0_12=data_FCD2_n[0:12,:]
chan13_18=data_FCD2_n[13:19,:]
chan20_26=data_FCD2_n[20:27,:]
chan28_38=data_FCD2_n[28:39,:]
chan41_62=data_FCD2_n[41:63,:]
chan65_66=data_FCD2_n[65:67,:]


data_FCD2_n=np.vstack((chan0_12, chan13_18,chan20_26,chan28_38,chan41_62,chan65_66))


################################ Data Normalization #######################

data_FCD2_n = preprocessing.normalize(data_FCD2_n)


##########################################################################

# This line is only for only patients 3

#chan0_36=data_FCD2_n[0:37,:]
#chan39_44=data_FCD2_n[39:44,:]

#data_FCD2_n=np.vstack((chan0_36, chan39_44))

        
for sg in range(180): # for test set we used 30 mins


    Temp_test=data_FCD2_n[:,s:f] 

    Frame1_test[sg,:,:]=Temp_test  

    s+=20000
    f+=20000

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

fs=1000 #Sampling frequency


segmented_frame=np.empty(shape=[60,20000])
stored_filtered_data=np.empty(shape=[60,7,20000])
seg_filtered_data=np.empty(shape=[180,60,7,20000])

for i in range(180): #number of segments
    segmented_frame=Frame1_test[i,:,:]
    for j in range(60):
        
        stored_filtered_data[j,:,:]=filter_bank(segmented_frame[j,:],fs)
        
    seg_filtered_data[i,:,:,:]=stored_filtered_data
        
        
    

 #seg_filtered_data=contain filterbank result   

#Apply new three feature on each subbands

Temp1= np.empty(shape=[7,9])       #subbans X features
Temp2=np.empty(shape=[60,7,9])     #channel X subbands X new_features
Temp3=np.empty(shape=[180,60,7,9])  #segments X cnannel X subbands X new_features

thres=0.016

for filtered_seg in range(180):
    q=np.squeeze(seg_filtered_data[filtered_seg,:,:,:])  
    print("Segment_{0}".format(filtered_seg))
    
    for ch in range(60):
        filtered_ch=np.squeeze(q[ch,:,:])
        
        
        for sub_band in range(7):
            
            Temp1[sub_band,:]=new_sir_feature(filtered_ch[sub_band,:])
            #print( Temp1.shape)
            
        Temp2[ch,:,:]=Temp1
        
    Temp3[filtered_seg,:,:,:]=Temp2
        
  
        
scipy.io.savemat('P10_sheuli_new_fea_0_180epoc.mat', mdict={'p10_sheufea': Temp3})        
    
