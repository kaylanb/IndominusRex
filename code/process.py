import pandas as pd
import numpy as n
import glob
from scipy.signal import butter, lfilter, freqz
from scipy.stats import kurtosis, skew, stats
import pickle

def get_filtered_data(dfile):
    data_tbl = pd.read_csv(dfile)
    events_tbl = pd.read_csv(dfile[:-8]+'events.csv')
    data_tbl.drop('id',axis=1,inplace=True) 
    events_tbl.drop('id',axis=1,inplace=True) 
    fdata= data_tbl.copy()
    fevents= events_tbl.copy()
    #get rid of pesky subject id that is repeated for N time points
    #low pass filter
    fs = 500 #Hz
    lowcut = 2 #Hz
    highcut = 8 #Hz
    for k in fdata.keys():
        fdata[k] = butter_bandpass_filter(fdata[k], lowcut, highcut, fs, order=4)
    return (fdata,fevents)

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

def index_event_beg_end(i_beg,i_end,series,event_name, data_for_event):
    index = 0
    while True:
        index_arr = n.where(data_for_event[index:] == 1)[0]
        if len(index_arr) == 0: break
        index+= index_arr[0]
        #ibeg is index of first 1, iend is index of last 1
        i_beg[series][event_name].append( index )
        i_end[series][event_name].append( index+149 )
        #set index to iend+1, so at a 0
        index+= 150 #width of 1's in event data
        
def ensemble_sum(ievent, buckets,fdata):
    for i in ievent: #4 buckets x 50 tpts x 32 channels
#         print i,buckets[0,:,:].shape,data_tbl.values.copy()[i-25:i+25,:].shape
#         print data_tbl.values.copy()[i-25:i+25,0]
        buckets[0,:,:]=buckets[0,:,:] + fdata.values.copy()[i-25:i+25,:]
        buckets[1,:,:]=buckets[1,:,:] + fdata.values.copy()[i+25:i+75,:]
        buckets[2,:,:]=buckets[2,:,:] + fdata.values.copy()[i+75:i+125,:]
        buckets[3,:,:]=buckets[3,:,:] + fdata.values.copy()[i+125:i+175,:]
        
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


def target_features_for_series(dfile,i_beg, verbose=None):
    '''returns: Target feature vector array for dfile series'''
    #get series data
    (data_tbl,events_tbl)= get_filtered_data(dfile)
    series= dfile[-16:-9]
    #feature vector array
    N_features= 10
    shape= (N_features,4*events_tbl.keys().shape[0]) #n features X 4 buckets*6 target events
    target_features= n.zeros(shape)
    target_event_type= n.zeros(4*events_tbl.keys().shape[0]).astype(int)-1
    #arr to store data: 4 buckets x 50 time pts x 32 channels x 6 events
    shape= (4,50,data_tbl.keys().shape[0],events_tbl.keys().shape[0]) 
    buckets= n.zeros(shape)
    #for each event, take average of all trials in this series
    for e,event in enumerate(events_tbl.keys()):
#         print "series,event,i_beg= ",series,event,i_beg[series][event]
        #add up all trials in the series
        ensemble_sum(i_beg[series][event], buckets[:,:,:,e],data_tbl)
        #divide by N trials so sum becomes average
        buckets[:,:,:,e]= buckets[:,:,:,e]/len(i_beg[series][event])   
    #average channels for each bucket and output feature vectors 
    cnt=0
    for e,event in enumerate(events_tbl.keys()):
        signal= n.mean(buckets[:,:,:,e],axis=2)
        for b in range(4):
            target_features[:,cnt]= get_features(signal[b,:])
            target_event_type[cnt]= e+1 #0 = non target (noise) events, 1-6 = target events
            cnt+=1
    if verbose is None: return (target_event_type,target_features)
    return (target_event_type,target_features,buckets,signal) #otherwise

def TEST_target_features_for_series(dfile,i_beg):
    (event_types,features,buckets,signal)= target_features_for_series(dfile,i_beg)
    fig,axis=plt.subplots(3,2)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    ax=axis.flatten()
    for e,event in enumerate(events_tbl.keys()):
        ax[e].plot(buckets[1,:,:,e])
    fig,axis=plt.subplots(2,2)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    ax=axis.flatten()
    for b in range(4):
        ax[b].plot(signal[b,:])
    for i in range(event_types.shape[0]): 
        print "event: ",event_types[i],", features: ",features[0:3,i]


def add_noise_windows(a,b,cnt_arr):
    '''a,b are numpy arrays with shape = n time points X 32 channels
    cnt_arr has shape: n time points and contains the number of times each index has been summed
        so averaging can be done
    n is different for a and b'''
    if a.shape[0] == max(a.shape[0],b.shape[0]): 
        c= a.copy()
        c[:b.shape[0],:]= c[:b.shape[0],:]+b.copy()
        new_cnt= n.zeros(a.shape[0]).astype('int')+1
    else: 
        c=b.copy()
        c[:a.shape[0],:]= c[:a.shape[0],:]+a.copy()
        new_cnt= n.zeros(b.shape[0]).astype('int')+1
    new_cnt[:len(cnt_arr)]= cnt_arr+1
    return (c,new_cnt)

def every_3rd_50pt_window(window):
    '''noise has huge window with varying number of time points
    returns: every 3rd 50 time point sub-window, as many sub-windows as there are in windows'''
    ind= n.arange(window.shape[0])[::50][::3]
    nwind= len(ind)-1
    shape= (nwind,50,window.shape[1]) #n windows x 50 time points x 32 channels 
    new_window= n.zeros(shape)
    for cnt,beg in zip(range(nwind),ind[:-1]):
        new_window[cnt,:,:]= window[beg:beg+50,:]
    return new_window

def avg_noise_BothReleased_HandStart(i_beg,i_end,series, data_tbl):
    #Average trials occuring in window: BothReleased to HandStart
    #sum signal in the trials
    window_1= data_tbl.values.copy()[:i_beg[series]['HandStart'][0], :]
    cnt_arr= n.zeros(window_1.shape[0]).astype(int)+1 #increments each index when index is used in a sum
    for ileft,iright in zip(i_end[series]['BothReleased'][:-1], i_beg[series]['HandStart'][1:]):
        (window_1,cnt_arr)= add_noise_windows(window_1, data_tbl.values.copy()[ileft+1:iright, :],cnt_arr)
    #finally, divide by count to get average
    for c,chan in enumerate(data_tbl.keys()):
        window_1[:,c]= window_1[:,c]/cnt_arr
    return every_3rd_50pt_window(window_1) #n windows x 50 time points x 32 channels 

def avg_noise_HandStart_FirstDigitTouch(i_beg,i_end,series, data_tbl):
    buf= 25 #otherwise have signal at min,max indices
    (ileft,iright)= (i_end[series]['HandStart'][0],i_beg[series]['FirstDigitTouch'][0])
    window_2= data_tbl.values.copy()[ileft+1+buf:iright-buf, :]
    cnt_arr= n.zeros(window_2.shape[0]).astype(int)+1
    for ileft,iright in zip(i_end[series]['HandStart'][1:], i_beg[series]['FirstDigitTouch'][1:]):
        (window_2,cnt_arr)= add_noise_windows(window_2, data_tbl.values.copy()[ileft+1+buf:iright-buf, :],cnt_arr)
    #divide to get average
    for c,chan in enumerate(data_tbl.keys()):
        window_2[:,c]= window_2[:,c]/cnt_arr
    return every_3rd_50pt_window(window_2)

def avg_noise_HandStart_FirstDigitTouch(i_beg,i_end,series, data_tbl):
    buf= 0 #otherwise have signal at min,max indices
    (ileft,iright)= (i_end[series]['LiftOff'][0],i_beg[series]['Replace'][0])
    window_3= data_tbl.values.copy()[ileft+1+buf:iright-buf, :]
    cnt_arr= n.zeros(window_3.shape[0]).astype(int)+1
    for ileft,iright in zip(i_end[series]['LiftOff'][1:], i_beg[series]['Replace'][1:]):
        (window_3,cnt_arr)= add_noise_windows(window_3, data_tbl.values.copy()[ileft+1+buf:iright-buf, :],cnt_arr)
    #divide to get average
    for c,chan in enumerate(data_tbl.keys()):
        window_3[:,c]= window_3[:,c]/cnt_arr
    return every_3rd_50pt_window(window_3)


def noise_features_for_series(dfile,i_beg,i_end, verbose=None):
    '''returns: Noise (non target) feature vector array for dfile series'''
    #get series data
    (data_tbl,events_tbl)= get_filtered_data(dfile)
    series= dfile[-16:-9]
    #feature vector array
    N_features= 10
    shape= (N_features,3) #n features X 3 noise windows
    noise_features= n.zeros(shape)
    noise_event_type= n.zeros(3).astype(int)-1
    #for each noise event, average trials, do PCA, and get features
    noise={}
    noise['0']= avg_noise_BothReleased_HandStart(i_beg,i_end,series, data_tbl)
    noise['1']= avg_noise_HandStart_FirstDigitTouch(i_beg,i_end,series, data_tbl)
    noise['2']= avg_noise_HandStart_FirstDigitTouch(i_beg,i_end,series, data_tbl)
    for i in [0,1,2]: noise[str(i)]= n.mean(noise[str(i)],axis=2) #avg over channels
    #get features
    nwindows= 0
    for i in [0,1,2]: nwindows+= noise[str(i)].shape[0]
    noise_features= n.zeros( (N_features,nwindows) )-1
    noise_event_type= n.zeros(nwindows).astype(int)-1
    cnt=0
    for i in [0,1,2]:
        for win in range(noise[str(i)].shape[0]): #n 50 pt windows
            noise_features[:,cnt]= get_features( noise[str(i)][win,:] )
            noise_event_type[cnt]= 0 #0 is noise event
            cnt+=1
    if verbose is None: return (noise_event_type,noise_features)
    return (noise_event_type,noise_features,noise) #otherwise
     

def TEST_noise_features_for_series(dfile,i_beg,i_end):
    (noise_event_type,noise_features,noise)= noise_features_for_series(dfile,i_beg,i_end, verbose=True)
    for i in range(len(noise_event_type)): 
        print "event: ",noise_event_type[i],", features: ",noise_features[0:3,i]
    fig,axis=plt.subplots(2,3)
    ax=axis.flatten()
    for win in range(6):
        ax[win].plot(noise['0'][win,:])
    fig,axis=plt.subplots(2,3)
    ax=axis.flatten()
    for win in range(6):
        ax[win].plot(noise['1'][win,:])
    fig,axis=plt.subplots(2,3)
    ax=axis.flatten()
    for win in range(6):
        ax[win].plot(noise['2'][win,:])


#MAIN
for subj in range(1,11):
	datafiles= glob.glob('../train/subj'+str(subj)+'_series*_data.csv')
	#i_beg is index of first 1, i_end is index of last 1
	i_beg={}
	i_end={}
	(data_tbl,events_tbl)= get_filtered_data(datafiles[0]) #to get event names
	#indices of target events in each series of trials
	for i in range(9):
		i_beg['series'+str(i)]={}
		i_end['series'+str(i)]={}
		for event in events_tbl.keys():
			i_beg['series'+str(i)][event]=[]
			i_end['series'+str(i)][event]=[]
	#feature vectors and event lables for each series of trials
	vector= {}
	for i in range(9):
		vector['series'+str(i)]= {}        
	#fill i_beg,i_end
	for dfile in datafiles:
		(data_tbl,events_tbl)= get_filtered_data(dfile)
		for event in events_tbl.keys():
			index_event_beg_end(i_beg,i_end,dfile[-16:-9],event, events_tbl[event]) 
	#get features and event labels
	for dfile in datafiles:
		print "loading file: %s" % (dfile)
		print 'extracting Target Features'
		(t_events,t_feats)= target_features_for_series(dfile,i_beg)
		print 'extracting Noise Features'
		(n_events,n_feats)= noise_features_for_series(dfile,i_beg,i_end)
		series= dfile[-16:-9]
		vector[series]['feats']= n.concatenate( (t_feats,n_feats),axis=1)
		vector[series]['events']= n.concatenate( (t_events,n_events),axis=1)
	#save feature vectors to pickle file
	tot_feats= n.concatenate((vector['series1']['feats'].copy(),\
			vector['series2']['feats'].copy(),vector['series3']['feats'].copy(),\
			vector['series4']['feats'].copy(),vector['series5']['feats'].copy(),\
			vector['series6']['feats'].copy(),vector['series7']['feats'].copy(),\
			vector['series8']['feats'].copy()),axis=1)
	tot_events= n.concatenate((vector['series1']['events'].copy(),\
			vector['series2']['events'].copy(),vector['series3']['events'].copy(),\
			vector['series4']['events'].copy(),vector['series5']['events'].copy(),\
			vector['series6']['events'].copy(),vector['series7']['events'].copy(),\
			vector['series8']['events'].copy()),axis=1)
	fname=datafiles[0][9:-4]+"_training_features.pickle"
	f= open(fname,"w")
	pickle.dump((tot_feats,tot_events),f)
	f.close()

