import tensorflow as tf
import numpy as np
import copy


checkpoints_folderName = 'Checkpoints'
checkpoints_modelName = 'model.ckpt'


##########################################################################################################
class RNN(object):
    g = None
    sess = None    
    # ---------------------------------------------------------------------------------------------------   
    def __init__(self, NNtype, Nlayers, Nunits, TBlength, Nclasses, Nfeatures, **optimizerParams):        
        self.__class__ = eval(NNtype)       
        self.NNtype = NNtype                # Type of RNN  
        self.Nlayers = int(Nlayers)         # Number of layers
        self.Nunits = int(Nunits)           # Number of units
        self.TBlength = int(TBlength)       # Truncated backpropagation length
        self.Nclasses = int(Nclasses)       # Number of classes
        self.Nfeatures = int(Nfeatures)     # Number of features     

        self.optimizerParams = optimizerParams       
    # ---------------------------------------------------------------------------------------------------   
    def build(self):
         if self.g is None: 
            if 'sess' in globals() and sess: sess.close() 
            tf.reset_default_graph() # reset graph
            self.g = dict()
            self.build_inputs()
            self.build_model()
            self.build_optimizer()
            self.reset()     
    # ---------------------------------------------------------------------------------------------------   
    def build_inputs(self):
        # Input placeholders
        self.g['inputs'] = tf.placeholder(tf.float32, shape=[None, self.TBlength, self.Nfeatures], name='inputs_placeholder')
        self.g['labels'] = tf.placeholder(tf.int32, shape=[None], name='labels_placeholder')
        self.g['keepProb'] = tf.placeholder(tf.float32, name='keepProb_placeholder')    
    # ---------------------------------------------------------------------------------------------------   
    def build_model(self):  
        # Batch normalization 
        inputs_norm = tf.contrib.layers.batch_norm(self.g['inputs'])  
        # Build Network
        net, init_state_tuple = self.get_net()    
        # RNN network output
        outputs, self.g['current_state'] = tf.nn.dynamic_rnn(net, inputs_norm, initial_state=init_state_tuple)       
        # Logits:  [batchSize, TBlength, Nunits] --> [batchSize, TBlength,  Nclasses] 
        with tf.variable_scope('Conv_mtx'):
            W = tf.get_variable('W', [self.Nunits, self.Nclasses])
            b = tf.get_variable('b', [1, self.Nclasses], initializer=tf.constant_initializer(0.0))                        
        logits = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, self.Nunits]), W) + b, [-1, self.TBlength, self.Nclasses])
        logits = tf.reduce_mean(logits,axis=1) # average over a window for logits         

        # Predictions
        probabilities = tf.nn.softmax(logits) # Sofmax 
        self.g['predictions'] = tf.cast(tf.argmax(probabilities,1),tf.int32)    
        
        # Accuracy
        equality = tf.equal( self.g['predictions'], self.g['labels'] )
        self.g['accuracy'] = tf.cast(equality,tf.float32)

        # Loss
        self.g['loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=self.g['labels'], logits=logits)         
    # ---------------------------------------------------------------------------------------------------   
    def build_optimizer(self):  
        # Optimizer
        optPars = copy.deepcopy(self.optimizerParams)
        optimizerType = optPars.pop('optimizerType')
        learningRate = optPars.pop('learningRate')
        globalStep = tf.Variable(0, trainable=False)  
        expDecaySteps = 100
        if 'expDecaySteps' in optPars: expDecaySteps = optPars.pop('expDecaySteps')
        expDecayRate = 1
        if 'expDecayRate' in optPars: expDecayRate = optPars.pop('expDecayRate')

        # decayed_learningRate = learningRate * decayRate ^ (globalStep / decaySteps)      
        decayed_learningRate = tf.train.exponential_decay(learningRate, globalStep, expDecaySteps, expDecayRate)       
        
        # Optimizer
        optPars.update({'learning_rate':decayed_learningRate})
        optimizer = eval( 'tf.train.' + optimizerType )( **optPars )        

        # Train
        loss_avg = tf.reduce_mean(self.g['loss'])        
        self.g['trainStep'] = optimizer.minimize(loss_avg, global_step=globalStep)      
    # ---------------------------------------------------------------------------------------------------   
    def train_validate( self, TrainValidation_data, isTraining=True, stride=None, batchSize=None, keepProb=1):
        # OUTPUTS:
        #         loss_list (vector of size Nepochs):  list of losses for all epochs 
        #         accuracy_list (vector of size Nepochs): list of accuracy for all epochs 

        mtx_data = TrainValidation_data.mtx()
        if batchSize is None: batchSize = mtx_data.X.shape[0]
        if stride is None: stride = self.TBlength 
        if isTraining: keepProb_ = keepProb
        else: keepProb_ = 1

        # Train / validate
        current_state = self.get_init_state(batchSize)
        ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list = list(), list(), list(), list()
        # results = RESULTS()                    
        for i in range( 0,  mtx_data.X.shape[0], batchSize ):
            for j in range( 0,  mtx_data.X.shape[1]-self.TBlength, stride ):
                batch_x = mtx_data.X[i:i+batchSize, j:j+self.TBlength]
                batch_y = mtx_data.Y[i:i+batchSize] 
                # Feeds
                feed_dict = { 
                    self.g['init_state']:current_state, 
                    self.g['inputs']:batch_x, 
                    self.g['labels']:batch_y, 
                    self.g['keepProb']:keepProb_
                    }
                # Outputs
                requested_outputs = [ 
                    self.g['current_state'], 
                    self.g['loss'],  
                    self.g['accuracy'], 
                    self.g['predictions']
                    ]
                if isTraining: requested_outputs.append(self.g['trainStep'])
                # Run session & get results
                outputs = self.sess.run(requested_outputs, feed_dict=feed_dict )                                  
                current_state, batch_loss_list, batch_accuracy_list, batch_prediction_list = outputs[:4]

                ep_loss_list.append(batch_loss_list)     
                ep_accuracy_list.append(batch_accuracy_list)
                ep_prediction_list.append(batch_prediction_list)
                ep_label_list.append(batch_y)
        
        ep_loss_list = np.array(ep_loss_list).flatten()
        ep_accuracy_list = np.array(ep_accuracy_list).flatten()
        ep_prediction_list = np.array(ep_prediction_list).flatten()
        ep_label_list = np.array(ep_label_list).flatten()

        return ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list    
    # ---------------------------------------------------------------------------------------------------   
    def run(self, training_data, validation_data, Nepochs=200, **trainingParams):
        
        training_results = dict({  'loss' : list(), 'accuracy' : list(), 'predictions': list(), 'labels' : list() })
        validation_results = dict({  'loss' : list(), 'accuracy' : list(), 'predictions': list(), 'labels' : list() })

        for ep in range( Nepochs ): 
            # Training
            ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list = self.train_validate(training_data, isTraining=True, **trainingParams)
            training_results['loss'].append(ep_loss_list)
            training_results['accuracy'].append(ep_accuracy_list)
            training_results['predictions'].append(ep_prediction_list)
            training_results['labels'].append(ep_label_list)

            # Validation                
            ep_loss_list, ep_accuracy_list, ep_prediction_list, ep_label_list = self.train_validate(validation_data, isTraining=False, **trainingParams)
            validation_results['loss'].append(ep_loss_list)
            validation_results['accuracy'].append(ep_accuracy_list)
            validation_results['predictions'].append(ep_prediction_list)
            validation_results['labels'].append(ep_label_list)
            
        return training_results, validation_results                     
    # ---------------------------------------------------------------------------------------------------       
    def reset(self):
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer()) # initialize the graph
    # ---------------------------------------------------------------------------------------------------       
    def save_model(self, folderPath):  
        # save checkpoints 
        checkpoints_folderPath = folderPath + '/' + checkpoints_folderName
        create_folderPath(checkpoints_folderPath) 
        saver = tf.train.Saver()       
        saver.save(self.sess, checkpoints_folderPath + '/' + checkpoints_modelName)        
    # ---------------------------------------------------------------------------------------------------       
    def restore_model(self, folderPath):
        # folderPath: '.../NNtype LSTM, ....Nepoch 10'
        checkpoints_folderPath = folderPath + '/' + checkpoints_folderName
        saver = tf.train.Saver()        
        saver.restore(self.sess, checkpoints_folderPath + '/' + checkpoints_modelName)                    
##########################################################################################################
class vanillaRNN(RNN):
    # ---------------------------------------------------------------------------------------------------       
    def get_net(self):
        net = []
        self.g['init_state'] = tf.placeholder(tf.float32, [self.Nlayers, None, self.Nunits], name='initState_placeholder')
        l = tf.unstack(self.g['init_state'], axis=0)
        init_state_tuple = tuple([l[idx] for idx in range(self.Nlayers)])
        for i in range(self.Nlayers): 
            cell = tf.contrib.rnn.BasicRNNCell(self.Nunits) 
            # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.g['keepProb'])  
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.g['keepProb'], output_keep_prob=self.g['keepProb'], state_keep_prob=self.g['keepProb'] )
            net.append(cell)
        net = tf.contrib.rnn.MultiRNNCell(net)          
        return net, init_state_tuple
    # ---------------------------------------------------------------------------------------------------       
    def get_init_state(self, batchSize):
        return np.zeros((self.Nlayers, batchSize, self.Nunits))
##########################################################################################################
class LSTM(RNN):
    # ---------------------------------------------------------------------------------------------------       
    def get_net(self):
        net = []
        self.g['init_state'] = tf.placeholder(tf.float32, [self.Nlayers, 2, None, self.Nunits], name='initState_placeholder')
        l = tf.unstack(self.g['init_state'], axis=0)    
        init_state_tuple = tuple([tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(self.Nlayers)])        
        for i in range(self.Nlayers): 
            cell = tf.contrib.rnn.BasicLSTMCell(self.Nunits, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.g['keepProb'])    
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.g['keepProb'], output_keep_prob=self.g['keepProb'], state_keep_prob=self.g['keepProb'] )
            net.append(cell)
        net = tf.contrib.rnn.MultiRNNCell(net, state_is_tuple=True)    
        return net, init_state_tuple
    # ---------------------------------------------------------------------------------------------------       
    def get_init_state(self, batchSize):
        return np.zeros((self.Nlayers, 2, batchSize, self.Nunits))
##########################################################################################################