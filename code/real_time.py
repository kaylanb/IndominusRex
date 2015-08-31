import pandas as pd
import numpy as n
import matplotlib.pyplot as plt
import glob
from scipy.signal import butter, lfilter, freqz
from scipy.stats import kurtosis, skew, stats
import pickle

def preprocess(dfile):
    data_tbl = pd.read_csv(dfile)
    fdata= data_tbl.copy()
    ids= fdata['id'].values
    #get rid of pesky subject id that is repeated for N time points
    fdata.drop('id',axis=1,inplace=True) 
    #low pass filter
    fs = 500 #Hz
    lowcut = 2 #Hz
    highcut = 8 #Hz
    for k in fdata.keys():
        fdata[k] = butter_bandpass_filter(fdata[k], lowcut, highcut, fs, order=4)
    return (fdata,ids)    

#low band pass filter all channels
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = .5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low,high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b,a = butter_bandpass(lowcut, highcut, fs, order=order)
    y=lfilter(b, a, data)
    return y

def get_features(bucket):
# feat[extra] = take std dev over some N pts > 30 then compute delta std / delta t
#     feat[extra]= sum_channels_over_all_series_for_same_event_type
#power spectrum of noise, first 2 dominant frequencies? -->need sample more than 50 time points
    feat=n.zeros(10)
    feat[0]= n.median(bucket)
    feat[1]= bucket.mean()
    feat[2]= bucket.min()
    feat[3]= bucket.max()
    feat[4]= n.std(bucket)
    feat[5]= skew(bucket)
    feat[6]= kurtosis(bucket)
    slope, intercept, r_value, p_value, std_err =         stats.linregress(n.arange(len(bucket)),bucket)
    feat[7]= slope
    feat[8]= r_value
    feat[9]= n.absolute(bucket).sum()
    return feat

def real_time_features(dfile):
    #get data
    (data_tbl,ids)= preprocess(dfile)
    n_points=50
    n_features= 10
    vec_feat= n.zeros((n_features,data_tbl.shape[0]))-1
    #get features
#     if tend == -1: tend= data_tbl.values.copy().shape[0]
    for t in range(50,data_tbl.values.copy().shape[0]):
        if t == 100 or t == 1000 or t == 10000 or t == 100000: print 't= ',t   
        vec_feat[:,t]= get_features( n.mean(data_tbl.values[t-n_points:t,:],axis=1) ) #50 time pts
    return (vec_feat,ids)

#MAIN
#datafiles= glob.glob('../test/*_data.csv')
#for dfile in datafiles:
for subj in range(1,13):
	dfile='../train/subj'+str(subj)+'_series6_data.csv'
	print 'loading %s' % dfile
	(feat,ids)= real_time_features(dfile)
	fname= dfile[9:-4]+"_Test_series6_features.pickle"
	f= open(fname,"w")
	pickle.dump((feat,ids),f)
	f.close()
	print 'saved pickle file'
