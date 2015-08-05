"""
Contains the Features class, which handles all of
 the feature creation and management for our project.
"""

import numpy as np
import multiprocessing as mp
from glob import glob

class Features():
    """
    Class that calculates all features.
    """

    def __init__(self, fname):
        self.fname = fname
        self.read_in_data()
        # find all of the features we want to include and set them up
        self.feature_funcs = []
        self.feature_names = []
        allfuncs = dir(self)
        allfuncs.sort()
        for f in allfuncs:
            if (f[0] == '_') and (f[1] != '_'):
                self.feature_funcs.append( getattr(self, f) )
                self.feature_names.append( f )
        # create the calculated features array
        self.features = np.zeros( len(self.feature_funcs) )
        # the time we're at in the data array
        self.t = 0
        
    def __iter__(self):
        # allows you to iterate over this object
        return self
    
    def next(self):
        """
        Calculate the next step of features and return them.
        """
        if self.t < (len(self.all_data)-1):
            self.update_features()
            return self.features
        else:
            raise StopIteration
        
    def update_features(self, t=None):
        """
        Updates all of the features.
        If given a timestep t, will go to that timestep.
        Otherwise updates self.t by one and calculates.
        """
        if t != None:
            self.t = t
        else:
            self.t +=1
        print 'Updating features to timestep',self.t
        self.current_data = self.all_data[:self.t]
        for i,f in enumerate(self.feature_funcs):
            self.features[i] = f()
    
    def read_in_data(self):
        """
        From Kaylan's code.
        """
        fin = open(self.fname, 'r')
        #get header info
        columns = fin.readline().split(',')
        columns[-1] = columns[-1].strip()
        fin.close()
        #read in all data
        formats = ['S10']+['f10']*(len(columns)-1)
        dtype = {'names': columns,'formats': formats}
        self.all_data = np.loadtxt(self.fname, delimiter=',', skiprows=1, dtype=dtype) 
        #time is number of EEG data points / 500, since 500 Hz sampling frequency
        self.all_time = np.arange(1, self.all_data['Fp1'].shape[0]+1)
    
    #################################################################
    # put feature functions here
    #
    # just add any function that starts with "_" and returns
    #  a single number, and it will be included in the feature array
    #################################################################
    
    def _feature1(self):
        # look at the 1000 most recent data points in the Fp1 array
        return np.mean( self.current_data['Fp1'][-1000:] )
    
    def _feature2(self):
        # look at the 100 most recent data points in the Fp1 array
        return np.mean( self.current_data['Fp1'][-100:] )
        
    
#################################################################
# this function calculates the feature vectors for a data file
#  and saves the result
#################################################################
def calc_features( fname ):
    F = Features( fname )
    allfeatures = np.array([feats for feats in F])
    np.savetxt( fname.replace('csv', 'feats.csv'), allfeatures, delimiter=',', header=','.join(F.feature_names) )
    print 'Calculated features from',fname,'and saved to',fname.replace('csv','feats.csv')

if __name__ == '__main__':
    #################################################################
    # calculate the features for all training data in parallel on 4 cores
    #################################################################
    files = glob( '../data/train/*data.csv' )
    pool = mp.Pool(4)
    pool.map_async( calc_features, files )

