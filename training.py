# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:29:56 2019

@author: Sanghyeon Lee
"""

import tensorflow as tf
import numpy as np
import random
import copy
import matplotlib.pyplot as plt


tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(777)

"""
# Tensorboard model save folder direction
import os
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logss')
if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)
"""

#Learning Parameters
learning_rate = 0.001

#Valuating time period
seq_length = 150

#Vd: Vm's gradient, Vm: normallized by standard MIXMAX, Irn: I rounded and normalized, Tn: T normalized, Pn: P noramlied
data_dim = 5 #Vd Vm Irn Tn Pn 

hidden_dim = 15         #RNN outputs which will be choosen by fully connected layer
output_dim = 1          #Y_pred's dimension
training_epochs = 100   #Number of learning

sec = 180               #Remaining time before 3.2V
Volt_max = 4.2          #Maximum Value of Voltage
Volt_min = 3.2          #Minimum Value of Voltage
downSize_div = 3        #Reducing size when OOM error happens.

#%% Data Preprocessing Define

# leng reducing sample size to avoid out of memory error
def sampleSizeDown(data, div):
    datad = []
    
    for i in range(0, int(len(data)/div)):#int(len(data)/div)):
        _d = data[i, :]
        datad.append(_d)
        
    return np.array(datad)

class preprocessing:
    sec = 180               #Remaining time before 3.2V
    Volt_max = 4.082        #Maximum Value of Voltage
    Volt_min = 3.2          #Minimum Value of Voltage
    downSize_div = 3        #Reducing size when OOM error happens.
    
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self._build_dataset()
        
    def _build_dataset(self):
        def power_append(xy):
            xy = np.c_[xy, np.zeros(int(len(xy)))]
            for i in range(int(len(xy))):
                xy[i][3] = xy[i][0]*xy[i][1]
            return xy    
            
        # If the voltage is decreased by 3.2V at t, Y label becomes '1' in time range [t-[sec], t].
        # The '1' in the Y label means warning signal that the battery will runout in [sec] seconds.
        def labeling(xy, sec):
            xy = np.c_[xy, np.zeros(int(len(xy)))]    
            for i in range(int(len(xy))):
                if (i >= sec):
                    xy[i][4] = 0
                    if(xy[i][0] <= 3.2):
                        i = i - sec
                        xy[i][4] = 1
                        for k in range(sec) :
                            i = i + 1
                            xy[i][4] = 1
            return xy
            
        # Normalize voltage value in 0~1, and regulate it into the range of [min_val, max_val]
        def VoltageMinMax(data, max_val, min_val):
            datav = data[:, 0:1]
            numerator = datav - np.min(datav,0)
            denominator = np.max(datav, 0) - np.min(datav, 0)
            datanv = numerator / (denominator + 1e-7)
            data[:, 0:1] = (datanv * (max_val - min_val)) + min_val
            return data
            
        # Change Voltage label to Vd label, Voltage difference between V(t-1) and V(t), (Vd, I, T, P)
        # This function is to analyze the importance of input variables.
        def chage_toVdelta(Xdata): 
            for i in range(int(len(Xdata)) - 1):
                Xdata[i][0] = (Xdata[i + 1][0] - Xdata[i][0])
            Xdata[i + 1][0] = Xdata[i][0]
            return Xdata
            
        # Add Vdelta label (Vd, V, I, T, P)
        def add_Vdelta(Xdata):
            delta_data = np.zeros((int(len(Xdata)), 1))
            delta_data = Xdata.copy()
            delta_data = np.insert(delta_data, 0, 0, axis=1)
            for i in range(int(len(Xdata)) - 1):
                delta_data[i][0] = (Xdata[i + 1][0] - Xdata[i][0])
            delta_data[i + 1][0] = delta_data[i][0]
            return delta_data
            
        # Rounding tiny difference in Current because when the current in a dataset is normalized,
        # tiny difference in current may make huge turbulence. 
        def rounding(Xdata):
            for i in range(int(len(Xdata))):
                Xdata[i][1] = round(Xdata[i][1],2) # I
            return Xdata
        
        self.power = power_append(self.data)
        self.label = labeling(self.power, sec)
        self.stablized = rounding(self.label)
        self.Vminmax = VoltageMinMax(self.stablized,Volt_max,Volt_min )
        self.addVd = add_Vdelta(self.Vminmax)
        
        return self.addVd

#%% Modify data formation to RNN model input form
        
class inputFormation:
    #Valuating time period
    seq_length = 150
    
    #Vd: Vm's gradient, Vm: normallized by standard MIXMAX, Irn: I rounded and normalized, Tn: T normalized, Pn: P noramlied
    data_dim = 5 #Vd Vm Irn Tn Pn 
    
    def __init__(self, data):
        self.data = data

    # Find the number of '1's and '0's in Y label
    # This variables can be used when we make a balanced dataset which is not a biased dataset with lots of '0's in Y label.
    def NG_count(self, xy, data_dim):
        j = 0
        for i in range(int(len(xy))):
            if (xy[i][data_dim] == 1):
                j = j + 1
        j0 = int(len(xy)) - j
        return  j , j0
    
    # Divide a dataset by seq_length size and accumulate it on 3rd dimension to make RNN input data form 
    # e.g. 100x5 (for timeseries: 100, # of inputs: 5) -> 10x(100-10+1)x5 (for seq_length = 10)
    def build_dataset(self, time_series, seq_length, data_dim):
        dataX = []
        dataY = []
        for i in range(0, len(time_series) - seq_length):
            _x = time_series[i:i + seq_length, 0:data_dim]
            _y = time_series[i + seq_length, data_dim:7]
            #print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)
        return np.array(dataX), np.array(dataY)
    
    # Combine different datasets
    def combine(self, xy1_x, xy1_y, xy2_x, xy2_y):
        dataX = np.vstack([np.array(xy1_x), np.array(xy2_x)])
        dataY = np.vstack([np.array(xy1_y), np.array(xy2_y)])
        
        return dataX, dataY
    
    
    # Scaling voltage, temperature, and power variables
    def Scaling_dataset(self, dataX): # dataX(7000,100,4)
        def MinMaxScalerbyRow(data):
            numerator = data - np.min(data, 0)
            denominator = (np.max(data, 0) - np.min(data, 0)) * 0.5
            MinMaxScaler = ((numerator) / (denominator + 1e-7)) - 1 # (1e-7: noise를 추가하여 분모가 0이 되는 것을 방지)
            return MinMaxScaler
        for i in range(dataX.shape[0]):
            dataX[i,:,2:5] = MinMaxScalerbyRow(dataX[i,:,2:5])
        return dataX

    
    # NG 값이 0인 데이터와 NG 값이 1인 데이터의 비율을 1:1 로 맞춘다.
    def balancing(self, Xdata, Ydata, Max):    
        databX = []
        databY = []
        j = 0
        k = 0
        leng = int(len(Ydata))
        randomRow = np.arange(leng) 
        random.shuffle(randomRow) # shuffle
        for i in randomRow:
            # 임의의 순번 중 NG가 0인 순번의 데이터를 NG1인 데이터의 수만큼 새롭게 쌓는다.
            if Ydata[randomRow[i],0] == 1:   
                if j < Max: 
                    _x1 = Xdata[randomRow[i],:]
                    _y1 = Ydata[randomRow[i],:]
                    databX.append(_x1)
                    databY.append(_y1)
                    j = j + 1
            else: 
                if k < Max:
                    _x2 = Xdata[randomRow[i],:]
                    _y2 = Ydata[randomRow[i],:]
                    databX.append(_x2)
                    databY.append(_y2)
                    k = k + 1
            if j >= Max and k >= Max:
                break
                
        return np.array(databX), np.array(databY)
    
    
    # 기존의 데이터의 값들을 무작위로 섞어 재배열
    def shuffle_dataset(self,dataX,dataY):
        shuffle = np.arange(dataY.shape[0])
        np.random.shuffle(shuffle)
        datasX = dataX[shuffle]
        datasY = dataY[shuffle]
        return datasX, datasY
    
    # Divide training data by 3, simple mini_batch
    def sampledivide(self, Xdata, Ydata):
        data1X = []
        data2X = []
        data3X = []
        data1Y = []
        data2Y = []
        data3Y = []
        leng = int(len(Ydata) / 3)
        for i in range(0, leng * 3):
            if i < leng:
                _1x = Xdata[i, :, :]
                _1y = Ydata[i, :]
                data1X.append(_1x)
                data1Y.append(_1y)
            elif (i >= leng) and (i < (leng * 2)):
                _2x = Xdata[i, :, :]
                _2y = Ydata[i, :]
                data2X.append(_2x)
                data2Y.append(_2y)
            else:
                _3x = Xdata[i, :, :]
                _3y = Ydata[i, :]
                data3X.append(_3x)
                data3Y.append(_3y)
                
        return np.array(data1X), np.array(data1Y), np.array(data2X), np.array(data2Y), np.array(data3X), np.array(data3Y)



#%%Model Architecture-------------------------------------------------------------------


class Model:
    '''
    To predict the battery pattern which can be seen in [sec] seconds before the battery runout,
    we used RNN architecture and studied optimal architectures
    
    We applied LSTMCell, multiRNNCell and dynamic_rnn through multiple tests.
    For loss function, we used cross entropy to minimize error by reducing local minimum error
    A predicted output becomes 1 when it's higher than 0.5.
    We can make prediction function by making output only 0 or 1.
    The accuracy rule is (TP+TF)/(TP+TN+FP+FN)
    
    '''
    
    def __init__(self, sees, name):
        self.sess = sess
        self.name = name
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            
            self.X = tf.placeholder(tf.float32, [None,seq_length, data_dim])
            
            #X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 1])
            
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hidden_dim, forget_bias=1.0)
                return lstm
            
            multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(5)])
            outputs, _states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype=tf.float32)
            
            Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
            self.hypothesis = tf.sigmoid(Y_pred)
            
            self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
            
        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))
        
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(self.cost)
                
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), tf.float32))
        
    def predict(self, x_test, training=False):
        return self.sess.run(self.hypothesis,feed_dict={self.X: x_test, self.training: training})
    
    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
                                                       self.Y: y_test, self.training: training})
    
    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

"""
Note: we removed Early Stopping because we found a feature that 
sometimes loss is increased but it is reduced again less than previous minimum loss
Thus, early stopping can hinder more accurate training 
"""

#%% MAIN
if __name__ == '__main__':
    #Data preprocessing---------------------------------------------------------------------------------
    #Preprocessing Order: loadtxt, power_append, Labeling -> V,I,T,P
    #   Naming Rule: DataUsage_FileNumber_PrepocessingSteps
    
    #Input data-----------------------------------------------------------------------------------------
    #TYPE1: Simple pattern Discharge Dataset [Voltage, Current, Temperture]
    #train_n1_s1 = np.loadtxt('RW25_refdis_1.csv', delimiter=',')
    #test_s1 = np.loadtxt('RW25_refdis_2.csv', delimiter=',')
    
    #TYPE2: Sophsticated pattern Charge and Discharge Dataset [Voltage, Current, Temperture]
    train_file = np.loadtxt('RW9_RW_1.csv', delimiter=',')
    test_file = np.loadtxt('RW9_RW_2.csv', delimiter=',')
    
    train_file = sampleSizeDown(train_file, downSize_div)
    
    #Data form modification
    dataset_A = preprocessing(train_file, "training_data")
    training_dataset = dataset_A._build_dataset()
    
    dataset_B = preprocessing(test_file, "test_data")
    test_dataset = dataset_B._build_dataset()
    
    #Change data form to RNN input form
    fm_A = inputFormation(training_dataset)
    j1_n1, j0_n1 = fm_A.NG_count(training_dataset, data_dim)        #Number 1 training data, NG label values(1,0) counting
    fm_B = inputFormation(test_dataset)
    k1, k0 = fm_B.NG_count(test_dataset, data_dim)              #Test data, NG label values(1,0) counting
                   
    
    #Training data transformation
    training_dataset = training_dataset[0:int(len(training_dataset))]      #Indexing
    train_built_X, train_built_Y = fm_A.build_dataset(training_dataset, seq_length, data_dim)     #Training Dataset building
    train_scaled_X = fm_A.Scaling_dataset(train_built_X)     #[Ir T P] labels normalizing to [Irn Tn Pn]
    
    #Test data transformation
    test_dataset = test_dataset[0:int(len(test_dataset))]                #Indexing
    test_built_X, test_built_Y = fm_B.build_dataset(test_dataset, seq_length,data_dim)           #Test Dataset building
    test_scaled_X = fm_B.Scaling_dataset(test_built_X)       #[Ir T P] labels normalizing to [Irn Tn Pn]
    
    #trainbX_1, trainbY_1 = balancing(train1X_1, train1Y_1, j1_1)
    
    """
    When we combine various dataset 
    trainc3X, trainc3Y = fm_A.combine(trainbX_1, trainbY_1,trainbX_3, trainbY_3)
    trainc5X, trainc5Y = fm_A.combine(trainc3X, trainc3Y, trainbX_5, trainbY_5)
    trainc7X, trainc7Y = fm_A.combine(trainbX_1, trainbY_1, trainbX_7, trainbY_7)
    trainc9X, trainc9Y = fm_A.combine(trainc7X, trainc7Y, trainbX_9, trainbY_9)
    trainc11X, trainc11Y = fm_A.combine(trainc9X, trainc9Y, trainbX_11, trainbY_11)
    trainc13X, trainc13Y = fm_A.combine(trainc7X, trainc7Y, trainbX_13, trainbY_13)

    Training sample sizing down when Out of Memory error occurred
    traindX, traindY = sampleSizeDown(trainsdX, train1Y_1, downSize)
    """
    #Randomly shuffled tree different Test dataset building
    #   balancing NG value size 1:1('0':'1') and randomly selecting data which has NG value '0'
    #   Shuffling their orders
    test_balanced_X_1, test_balanced_Y_1 = copy.deepcopy(fm_B.balancing(test_scaled_X, test_built_Y, k1))
    test_balanced_X_2, test_balanced_Y_2 = copy.deepcopy(fm_B.balancing(test_scaled_X, test_built_Y, k1))
    test_balanced_X_3, test_balanced_Y_3 = copy.deepcopy(fm_B.balancing(test_scaled_X, test_built_Y, k1))
    testX_1, testY_1 = fm_B.shuffle_dataset(test_balanced_X_1, test_balanced_Y_1)
    testX_2, testY_2 = fm_B.shuffle_dataset(test_balanced_X_2, test_balanced_Y_2)
    testX_3, testY_3 = fm_B.shuffle_dataset(test_balanced_X_3, test_balanced_Y_3)
    
 

    #Training Monitor
    with tf.Session() as sess:
     
        models = []
        rec_acc_list = []
        rec_cost_list = []
        num_models = 5      #ensamble set size
        
        for m in range(num_models):
            models.append(Model(sess, "model" + str(m)))
            
        sess.run(tf.global_variables_initializer())   #initialization
        
        print('Learning Start!')
        
        for epoch in range(training_epochs):
            avg_cost_list = np.zeros(len(models))
            
            #shuffle training input in every epoch because balancing function selected partical dataset
            train_balanced_X_1, train_balanced_Y_1 = fm_A.balancing(train_scaled_X, train_built_Y, j1_n1)    
            train_shuffled_X, train_shuffled_Y = fm_A.shuffle_dataset(train_balanced_X_1,train_balanced_Y_1)
            train_div1_X, train_div1_Y, train_div2_X, train_div2_Y, train_div3_X, train_div3_Y =  fm_A.sampledivide(train_shuffled_X, train_shuffled_Y)
            
            for m_idx, m in enumerate(models):
                c1, _ = m.train(train_div1_X, train_div1_Y)
                c2, _ = m.train(train_div2_X, train_div2_Y)
                c3, _ = m.train(train_div3_X, train_div3_Y)
                avg_cost_list[m_idx] += (c1+c2+c3) / 3
            ra = m.get_accuracy(testX_1,testY_1)
            rec_acc_list.append(ra)
            rec_cost_list.append(avg_cost_list)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
        
        print('Learning Finished!')#------------------------------------------------------------------------
        
        test_size = len(testY_1)
        predictions = np.zeros([test_size, int(len(models))])
        for m_idx, m in enumerate(models):
            print(m_idx, 'Accuracy:', m.get_accuracy(
                    testX_1, testY_1))
            p = m.predict(testX_1)
            for i in range(test_size):
                    predictions[i,m_idx] = p[i]
        tensorPredic = tf.constant(predictions)
        bestCase = tf.reduce_max(tensorPredic, reduction_indices=[1])
        pred = tf.cast(bestCase > 0.5, dtype=tf.float32)
        xxx = tf.reshape(pred, shape=[test_size,-1]).eval(session=sess)
        ensemble_correct_prediction = tf.equal(xxx,testY_1)
        ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
        
        print('Ensemble accuracy:', sess.run(ensemble_accuracy))  
        
    # Plot predictions
    plt.figure(1)
    plt.xlim([0, training_epochs])
    plt.title('Learning record')
    plt.plot(rec_cost_list,'-r',label='LOSS_Train')
    plt.plot(rec_acc_list,'-g',label='Accuracy_Test')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss / Accuracy")
