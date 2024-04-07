import numpy as np
import scipy as sp
import os
from scipy.signal import butter, lfilter, filtfilt
from mne import io
from entropy_lib.entropy import *
import sys
import datetime
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler


#################################################################
###                                                           ###
###  When changing EDF files, only rewrite here (3 parameters) ###
###                                                           ###
#################################################################
EDF_PATH = '/Volumes/ExternalHDD/intracranial/FCD/00_out_201811_EEG_11618421/'   # This is the path of data of data in Juntendo server
EDF_NAME = EDF_PATH +'DJ0010MV.edf'                      #This is for Patient 1
#channel_array = np.array([i for i in range(60)]) ### chane N of Channels

channel_array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76])


SAVE_DIR = './FCD6_one_hour_3ent'                     #This is for Patient 1
#channel_array = np.array([i for i in range(60)]) ### chane N of Channels


#data_FCD = io.read_raw_edfEDF_NAME, preload=True, stim_channel=None)
data_FCD = io.read_raw_edf(EDF_NAME, preload=True, stim_channel=None) ### change FILE NAME

data_suu = data_FCD[0][0].shape[0]

n_channels = channel_array.shape[0]

time_span = 40000 #20秒

section = 20
new_fs = 2000
#analysys_section = 30 #minutes
analysys_section = 60    #for 1 hour
a_s = analysys_section * 3
time_span = new_fs * section
#M = 2
M = 2
R = 0.2

####################
#fast ripple
#フィルタ後の信号作成
fs = 2000.0
#lowcuts = [100+50*i for i in range(3)]
#highcuts = [150+50*i for i in range(3)]

lowcuts = [100+50*i for i in range(10)]
highcuts = [150+50*i for i in range(10)]


#ENTROPY_NAMES = ["ShEn", "RenEn", "PE", "SpEn", "ApEn", "phEnS1", "phEnS2", "TsEn"]
ENTROPY_NAMES = ["ShEn", "RenEn", "PE"]

#ENTROPY_NAMES = ["SpEn", "ApEn", "phEnS2"]

#フィルタ
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

def entropys_for_each_epochs(args):
    epoch_idx, band_idx, ch_idx, std, sliced_dat = args
    print('band_idx:{}, ch_idx:{}/{}, time:{}/{} [{}]'.format(band_idx, ch_idx+1, n_channels, epoch_idx+1, a_s, datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    ShEn, RenEn = spectral_entropy(sliced_dat)
    PE = permutation_entropy(sliced_dat)
    #S1, S2 = bispectral_entropy(sliced_dat)
    #SpEn = samp_entropy(sliced_dat, M, R, std)
    #ApEn = ap_entropy(sliced_dat, M, R, std)
    #TsEn = tsallis_entropy(sliced_dat, 2)
    return np.array([ShEn, RenEn, PE])

def entropys_for_all_epochs(ch_idx, channel, band_idx, lowcut, highcut):
    iEEG = data_FCD[channel][0]
    iEEG = iEEG[0]
    #フィルタを通す
    filted_iEEG = butter_bandpass_filter(iEEG, lowcut, highcut, fs, order=3)
    if highcut != 600: filted_iEEG = filted_iEEG[::2]
    std = filted_iEEG.std()
    p = Pool(32)
    args = [(ep_idx, band_idx, ch_idx, std, filted_iEEG[ep_idx * time_span: ep_idx * time_span + time_span]) for ep_idx in range(a_s)]
    results = p.map(entropys_for_each_epochs, args) # results shape is (epochs, n_entropys)
    p.close()
    return np.array(results)

def entropys_for_all_channels(band_idx, lowcut, highcut):

    results = np.empty((0, a_s, len(ENTROPY_NAMES)))
    for ch_idx, channel in enumerate(channel_array):
        res = entropys_for_all_epochs(ch_idx, channel, band_idx, lowcut, highcut)
        results = np.append(results, [res], axis=0)

    for idx, en in enumerate(ENTROPY_NAMES):
        ### change FCD1_En to FCDn_En
        #np.savetxt('./FCD4_En/'+ '_'+ str(lowcut) + "_" + str(highcut) + "_" + 'pt04_' + en + '.csv', results[:,:,idx].T, delimiter=',')
        np.savetxt(SAVE_DIR + '/'+ str(lowcut) + "_" + str(highcut) + "_" + en + '.csv', results[:,:,idx].T, delimiter=',')


if __name__ == '__main__':
    #if not os.path.isdir("./FCD4_En"):
    ### change FCD1_En to FCDn_En
    if not os.path.isdir(SAVE_DIR):       #FCD1=saving directoty
        #os.mkdir("./FCD4_En")
        ### change FCD1_En to FCDn_En
        os.mkdir(SAVE_DIR)

    for band_idx, (lowcut, highcut) in enumerate(zip(lowcuts, highcuts)):
        entropys_for_all_channels(band_idx, lowcut, highcut)
