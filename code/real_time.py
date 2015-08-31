import argparse
parser = argparse.ArgumentParser(description="test")
parser.add_argument("-dfile",action="store",help='standard out/err from simulation')
parser.add_argument("-tbeg",type=int,action="store",help='text file with local rmsV,rmsVa,meanRho around each sink')
parser.add_argument("-tend",type=int,action="store",help='text file with local rmsV,rmsVa,meanRho around each sink')
args = parser.parse_args()


import pandas as pd
import numpy as n
import glob
from scipy.signal import butter, lfilter, freqz
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from scipy.stats import kurtosis, skew, stats

def get_data(dfile):
    data_tbl = pd.read_csv(dfile)
    ids= data_tbl['id'].values
    #get rid of pesky subject id that is repeated for N time points
    data_tbl.drop('id',axis=1,inplace=True) 
    return (data_tbl,ids)

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
    #bucket has shape (50,2) = (time points, pca components)
    #each bucket as 2 time series, 0th is best PCA, 1st is 2nd best PCA
    #features are diff in area between 0th and 1st PCA, then 0th PCA stats, then 1st PCA stats
    feat=n.zeros(17)
# feat[extra] = take std dev over some N pts > 30 then compute delta std / delta t
#     feat[extra]= sum_channels_over_all_series_for_same_event_type
#power spectrum of noise, first 2 dominant frequencies? -->need sample more than 50 time points
    feat[0]= (n.absolute( bucket[:,0] ).sum() - n.absolute( bucket[:,1] ).sum() )/len(bucket[:,0])
    #best PCA
    feat[1]= n.absolute( bucket[:,0] ).sum()/len(bucket[:,0]) #noise data has more time points than event data
    feat[2]= bucket[:,0].min()
    feat[3]= bucket[:,0].max()
    feat[4]= n.std(bucket[:,0])
    feat[5]= skew(bucket[:,0])
    feat[6]= kurtosis(bucket[:,0])
    slope, intercept, r_value, p_value, std_err =         stats.linregress(n.arange(len(bucket[:,0])),bucket[:,0])
    feat[7]= slope
    feat[8]= r_value
    #2nd best PCA
    feat[9]= n.absolute( bucket[:,1] ).sum()/len(bucket[:,1])
    feat[10]= bucket[:,1].min()
    feat[11]= bucket[:,1].max()
    feat[12]= n.std(bucket[:,1])
    feat[13]= skew(bucket[:,1])
    feat[14]= kurtosis(bucket[:,1])
    slope, intercept, r_value, p_value, std_err =         stats.linregress(n.arange(len(bucket[:,1])),bucket[:,1])
    feat[15]= slope
    feat[16]= r_value
    return feat

def real_time_features(dfile,tbeg,tend):
	#get data
	(data_tbl,ids)= get_data(dfile)
	n_points=50
	n_features= 17
	#will append features to this file
	f=open(dfile[8:-4]+"_features.csv",'a')
	#general format string for output to file
	line='%s'
	for i in range(n_features): line=line+ ',%.2f'
	line=line+'\n'
	#     f.write('id,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17\n')
	#get features
	feat= n.zeros((n_features,data_tbl.shape[0]))-1
	if tend == -1: tend= data_tbl.values.copy().shape[0]
	for t in range(tbeg,tend): #data_tbl.values.copy().shape[0]):
		if t == 100 or t == 1000 or t == 10000 or t == 100000: print 't= ',t
		curr_data= data_tbl.values.copy()[t-n_points:t,:]
		#low pass filter
		fs = 500 #Hz
		lowcut = 2 #Hz
		highcut = 8 #Hz
		for c,chan in enumerate(data_tbl.keys()):
			curr_data[:,c] = butter_bandpass_filter(curr_data[:,c], lowcut, highcut, fs, order=4)
		#PCA to reduce channel dimension from 32 to 2
		pca = PCA(n_components=data_tbl.keys().shape[0])
		K= pca.fit_transform( curr_data ) #arg shape: N time points X n channels
		a= get_features(K[:,:2]) #50 time pts, 2 pca components
		#append features to file
		f.write(line % (ids[t-1],a[0],a[1],a[2],a[3],a[4],a[5],               a[6],a[7],a[8],a[9],a[10],a[11],               a[12],a[13],a[14],a[15],a[16]))
	f.close()

#MAIN
real_time_features(args.dfile,args.tbeg,args.tend)
