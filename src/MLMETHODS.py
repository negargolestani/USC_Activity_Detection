import numpy as np

from sklearn import*
from itertools import chain
import matplotlib.pyplot as plt

eps = 1e-12


##############################################################################################################
class Bag_of_Words(object):
    # ---------------------------------------------------------------------------------------------------   
    def __init__(self, Ncodewords=20, training_features=None):
        self.model = cluster.KMeans(n_clusters=Ncodewords)
        if training_features is not None: 
            self.set_codebook(training_features)
    # ---------------------------------------------------------------------------------------------------   
    def minmax(self, X):
        return (X - self.min) / ( self.max - self.min + eps)
    # ---------------------------------------------------------------------------------------------------   
    def set_codebook(self, training_features):        
        Y = list()
        for idx, y in enumerate(training_features.Y):             
            Y.append( [y] * len(training_features.X[idx]) )  
        Y = list(chain.from_iterable( Y ))         
        X = list(chain.from_iterable( training_features.X ))   
        
        self.min = np.nanmin(X, axis=0)
        self.max = np.nanmax(X, axis=0)
        self.model.fit( self.minmax(X), Y)
        return 
    # ---------------------------------------------------------------------------------------------------   
    def get_codewords( self, X):
        codewords = np.zeros( (len(X), self.model.n_clusters) )
        for i, f in enumerate(X):
            for c in self.model.predict( self.minmax(f) ): 
                codewords [i, c] +=1
        return np.array( codewords ) 
##############################################################################################################


##############################################################################################################
class CLASSIFIER(object):
    # ---------------------------------------------------------------------------------------------------   
    def __init__(self, model, grid_params ):
        self.model = model
        self.grid_params = grid_params
    # ---------------------------------------------------------------------------------------------------   
    def train(self, training_data):
        clf = model_selection.GridSearchCV( self.model, self.grid_params )
        clf.fit( training_data.X, training_data.Y )
        self.model = clf.best_estimator_
        self.best_params = clf.best_params_
    # ---------------------------------------------------------------------------------------------------   
    def evaluate(self, validation_data, doShow=True):
        predictions = self.model.predict( validation_data.X )

        # Confusion Matrix
        labels = validation_data.Y 
        L = np.unique(labels) 
        conf_mtx = np.zeros(( len(L), len(L) ))
        for i, y in enumerate(L):
            for j, y_hat in enumerate(L):
                conf_mtx[i,j] = sum( (labels==y)*(predictions==y_hat) )  
            conf_mtx[i,:] = np.round( conf_mtx[i,:]/ sum(conf_mtx[i,:]) *100, 1)               
        accuracy = np.round( np.mean( labels == predictions ) *100, 1)
        precision = np.round( np.mean(conf_mtx.diagonal() / np.sum(conf_mtx, axis=0)) *100, 1 )
        recall = np.round( np.mean(conf_mtx.diagonal() / np.sum(conf_mtx, axis=1)) *100, 1) 
        F1 = np.round(2 * (precision * recall) / (precision + recall) /100, 2) 

        validation_results = dict(
            predictions = predictions,
            labels = labels,
            conf_mtx = conf_mtx,
            accuracy = accuracy,
            precision = precision,
            recall = recall,
            F1 = F1
        )

        if doShow:
            print( 'Best Parameters:', classifier.best_params)            
            print( 'Accuracy =', accuracy, '%' )
            print( 'Precision =', precision, '%' )
            print( 'Recall =', recall, '%' )
            print( 'F1 =', F1 )
            fig, axs = plt.subplots()
            the_table = axs.table( cellText=conf_mtx, cellLoc='center', loc='center')        
            for i in range(conf_mtx.shape[0]):
                for j in range(conf_mtx.shape[1]):
                    if conf_mtx[i,j] > 0:
                        the_table[(i,j)].set_facecolor("silver")
            the_table.scale(1.4,6)
            plt.axis('off')
            plt.show()
        else:
            return validation_results
    # ---------------------------------------------------------------------------------------------------   
##############################################################################################################
