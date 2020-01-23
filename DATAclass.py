import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pywt

from itertools import chain
from scipy import stats, signal
from sklearn import*

eps = 1e-12

##########################################################################################################
class DATA(object):    
    # ---------------------------------------------------------------------------------------------------   
    def __init__(self, filePath=None, X=None, Y=None, S=None, activity_dict=None):    
        if filePath is not None: 
            self.load(filePath)
        else:
            if X is not None: self.X = copy.deepcopy(X)
            if Y is not None: self.Y = copy.deepcopy(Y)
            if S is not None: self.S = copy.deepcopy(S)
            if activity_dict is not None: self.activity_dict = copy.deepcopy(activity_dict)   
        return
    # ---------------------------------------------------------------------------------------------------   
    def load(self, filePath):   
        self.X = list()
        self.Y = list()
        self.S = list()
        self.activity_dict = dict()
    
        database = np.load(filePath)
        for data in database:
            x = data['data']    
            s = data['subjectNum']
            activity = data['activityType']
            if activity in self.activity_dict:
                y = self.activity_dict[activity]
            else:
                y = len(self.activity_dict)
                self.activity_dict.update({activity:y})

            self.X.append(x)
            self.Y.append(y)
            self.S.append(s)
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.S = np.array(self.S)
        return
    # ---------------------------------------------------------------------------------------------------   
    def mtx( self, Nt_mtx='max' ):  
        # This function padds or cuts all input data (X) to make them same length and generate matrix data(X_mtx)
        # it also nomalize data X-mean(X)
        data_mtx = copy.deepcopy(self)
        if len(np.shape(data_mtx.X))>1:  return data_mtx    

        Nd, Nf = len(self.X),  np.shape(self.X[0])[1]
        Nt_list = list()
        for x in self.X: Nt_list.append( np.shape(x)[0] )
        if type(Nt_mtx) is str: Nt = int( eval('np.' + Nt_mtx)(Nt_list) )
        else:  Nt = Nt_mtx
        data_mtx.X = np.zeros( (Nd,Nt,Nf) )
        for idx, x in enumerate(self.X): 
            # x = np.subtract(x,np.mean(x,axis=0))        
            nt = np.shape(x)[0]
            if Nt >= nt:
                data_mtx.X[idx,:,:] = np.pad( x, ((0,Nt-nt),(0,0)),'constant')
            else:
                data_mtx.X[idx,:,:] = x[:Nt,:]
        return data_mtx
    # ---------------------------------------------------------------------------------------------------   
    def bound(self, min_value=None, max_value=None):
        # This function limits the amplitude value 
        
        bounded_data = copy.deepcopy(self)
        if min_value is not None:
            for x in bounded_data.X: x[ x<min_value ] = min_value
        if max_value is not None:                
            for x in bounded_data.X: x[ x>max_value ] = max_value
        
        return bounded_data
    # ---------------------------------------------------------------------------------------------------   
    def trim(self, keep_ratio=None):
        trimmed_data = copy.deepcopy(self)
        trimmed_data.X = list()
        
        if keep_ratio is None:
            dt = 20   
            for x in self.X:     
                N = len(x)
                n1, n2 = dt, N-dt 
                xx = abs( np.diff(x))
                xx = np.sum(xx, axis=1)    
                xx = abs(np.diff(xx))
                xx /= ( np.nanmax(xx) + eps )                 
                idxs = np.where( xx > 0.5 )[0]    
                idxs1 = idxs[idxs < 0.5*N] 
                idxs2 = idxs[idxs > 0.5*N]      
                if np.any(idxs1): n1 = np.min(idxs1) + dt
                if np.any(idxs2): n2 = np.max(idxs2) - dt   
                if (n2-n1) < 0.5*N: n1, n2 = 0, N            
                trimmed_data.X.append( x[n1:n2,:] )
        else:   
            for x in self.X:
                L = int( len(x) * keep_ratio)
                trimmed_data.X.append( x[:L,:] ) 

        trimmed_data.X = np.array(trimmed_data.X)    
        return trimmed_data    
    # ---------------------------------------------------------------------------------------------------      
    def quantize(self, Qstep):        
        quantized_data = copy.deepcopy(self)
        for idx, x in enumerate(quantized_data.X): 
            quantized_data.X[idx] = Qstep * np.floor(x/Qstep)
        return quantized_data   
    # ---------------------------------------------------------------------------------------------------   
    def clean(self):
        # cleans data from NANs ! 
        cleaned_data = copy.deepcopy(self)
        for idx, x in enumerate(cleaned_data.X):
            if np.any(np.isnan(x)):
                df = pd.DataFrame(x)
                df.fillna(method='ffill', axis=0, inplace=True)
                cleaned_data.X[idx] = df.as_matrix()
                
        return cleaned_data                
    # ---------------------------------------------------------------------------------------------------   
    def filter_noise(self, window_length=5, polyorder=2):
        filtered_data = copy.deepcopy(self)
        for n, x in enumerate(self.X):
            for i in range(8):
                filtered_data.X[n][:,i] = signal.savgol_filter(x[:,i], window_length, polyorder)        
        return filtered_data
    # ---------------------------------------------------------------------------------------------------   
    def MinMax(self):
        # Rescale data value to (0,1)
        normalized_data = copy.deepcopy(self)
        for idx, x in enumerate(normalized_data.X): 
            MIN = np.nanmin(x,axis=0)
            MAX = np.nanmax(x,axis=0)
            normalized_data.X[idx] = np.subtract(x,MIN) / ( np.subtract(MAX,MIN) + eps )
        return normalized_data    
    # ---------------------------------------------------------------------------------------------------    
    def standardize(self, scale=True):
        normalized_data = copy.deepcopy(self)
        STD = 1
        for idx, x in enumerate(normalized_data.X): 
            MEAN = np.mean(x,axis=0)
            if scale: STD = np.std(x,axis=0) + eps
            normalized_data.X[idx] = np.subtract(x,MEAN) / STD    
        return normalized_data         
    # ---------------------------------------------------------------------------------------------------   
    def summary(self):     
        min_l, max_l = np.inf, 0
        for x in self.X:
            x_l = np.shape(x)[0]
            min_l = min(min_l,x_l)
            max_l = max(max_l,x_l)
        textInfo = '\n Activities: '
        for activity_key, activity_value in self.activity_dict.items():
            textInfo += '\n \t    ('+ activity_key +' --> ' + str(activity_value) + ')'
            textInfo += ' (' + str(len(np.flatnonzero(self.Y == activity_value))) + ' samples)'    
        textInfo += '\n'
        textInfo += '\n Number of samples = ' + str(len(self.X))
        textInfo += '\n Number of subjects = ' + str(len(np.unique(self.S)))
        textInfo += '\n Number of features = ' + str(np.shape(self.X[0])[1])
        textInfo += '\n Min. data length = ' + str(min_l)
        textInfo += '\n Max. data length = ' + str(max_l)
        print(textInfo)
    # ---------------------------------------------------------------------------------------------------   
    def show(self, activity=0, Nsamples=1):    
        if type(activity) is str:
            y = self.activity_dict[activity]
        else:
            y = activity
            
        idx_list = np.where(self.Y==y)[0]
        for idx in list( idx_list[:Nsamples] ):
            f = plt.figure()
            plt.rcParams.update({'font.size': 20})
            plt.plot(self.X[idx], linewidth=3 )
            label_ID = self.Y[idx]
            label_str = list(self.activity_dict.keys()) [list(self.activity_dict.values()).index(label_ID)]
            plt.title(label_str)   
    # ---------------------------------------------------------------------------------------------------         
    def remove(self, remove_idx):
        modified_data = copy.deepcopy(self)
        modified_data.X = np.delete( modified_data.X, remove_idx ) 
        modified_data.Y = np.delete( modified_data.Y, remove_idx ) 
        modified_data.S = np.delete( modified_data.S, remove_idx ) 
        return modified_data
    # ---------------------------------------------------------------------------------------------------         
    def LOSO(self, training_subjNum=None, validation_subjNum=None):
        
        IDX = set( np.arange( len(self.X) ) )
        training_idx, validation_idx = list(), list()

        if training_subjNum is not None:
            for s in training_subjNum: training_idx = [ *training_idx, *np.where(self.S == s)[0] ]
            if validation_subjNum is None: validation_idx = list( IDX - set(training_idx) )  

        if validation_subjNum is not None:
            for s in validation_subjNum: validation_idx = [ *validation_idx, *np.where(self.S == s)[0] ]
            if training_subjNum is None: training_idx = list( IDX - set(validation_idx) )  

        training_data = self.remove(validation_idx)
        validation_data = self.remove(training_idx)

        return training_data, validation_data  
    # ---------------------------------------------------------------------------------------------------
    def get_features(self, TBlength=None, stride=None, Ns=None):
        if stride is None: stride= TBlength

        features = copy.deepcopy(self)
        features.X = list()
        
        for idx, x in enumerate(self.X): 
            f = list()
            Nx = x.shape[0] 

            if Ns is not None:
                TBlength = int(np.ceil(Nx/Ns))
                stride = TBlength

            for t in range(0, Nx-10, stride): 
                f.append( featureExtraction_segment(x[ t : min(Nx, t+TBlength), :]) )
                
            features.X.append( np.array(f) )

        features.X = np.array(features.X)
        features = features.clean() 
        return features
    # ---------------------------------------------------------------------------------------------------         
##############################################################################################################


##############################################################################################################
def featureExtraction_segment( x_t ):
    feature = list()   
    x_f = np.real( np.fft.fft(x_t, axis=0) )
    x_wA, x_wD = pywt.dwt(x_t, 'db1', axis=0)
    dx_t = np.diff( x_t, axis=0 )
    
    for x_ in [x_t, x_f, x_wA, x_wD, dx_t]:                                    
        feature.append( np.mean( x_, axis=0 )) 
        feature.append( np.std( x_, axis=0 ))                               
        feature.append( np.median( x_, axis=0 ))               
        feature.append( np.min( x_, axis=0 ))               
        feature.append( np.max( x_, axis=0 ))               
        feature.append( np.var( x_, axis=0 ))               
        feature.append( np.percentile( x_, 25, axis=0 ))               
        feature.append( np.percentile( x_, 75, axis=0 ))               
        feature.append( stats.skew( x_, axis=0))               
        feature.append( stats.kurtosis( x_, axis=0))               
        feature.append( stats.iqr( x_, axis=0))               
        feature.append( np.sqrt(np.mean(np.power(x_,2), axis=0)))               

        corr_t = np.corrcoef(np.transpose(x_))               
        for n in range(len(corr_t)-1):
            feature.append( corr_t[n, n+1:] )
        return np.array( list(chain.from_iterable(feature)) )
##############################################################################################################

