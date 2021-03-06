import pandas as pd
import numpy as np
import tensorflow as tf

import sys

def LoadData():
    pd_data = pd.read_csv('WithdrawalPredictionData.csv' ,index_col = 0, header = 0)
    #print(pd_data)
    return pd_data
    
def MakeTrainingData_x(pd_data):
    pd_data_dn = pd_data.dropna()
    #print(pd_data_dn)
    select = [0,1,2,3,4,5,6,7,8,9,10]
    pd_data_dn_select = pd_data_dn[select]
    #print(pd_data_dn_select)
    pd_data_dn_select_norm = pd_data_dn_select.apply(lambda x: (x/x.std()), axis=0).fillna(0)
    #print(pd_data_dn_select_norm)
    np_data_x = pd_data_dn_select_norm.values
    #print(np_data_x)
    np.savetxt("TrainingData_x.csv", np_data_x, delimiter=",")
    return np_data_x

def MakeTrainingData_y(pd_data):
    pd_data_dn = pd_data.dropna()
    #print(pd_data_dn)
    select = [11,12]
    pd_data_dn_select = pd_data_dn[select]
    #print(pd_data_dn_select)
    np_data_y = pd_data_dn_select.values
    #print(np_data_y)
    np.savetxt("TrainingData_y.csv", np_data_y, delimiter=",")
    return np_data_y

def Training(input_num,hidden_1_num,hidden_2_num,output_num,test_size,np_data_x,np_data_y,training_times):
    INPUT = input_num
    #print(INPUT)
    HIDDEN_1 = hidden_1_num
    #print(HIDDEN_1)
    HIDDEN_2 = hidden_2_num
    #print(HIDDEN_2)
    OUTPUT = output_num
    #print(OUTPUT)
    TEST_SIZE = test_size
    #print(TEST_SIZE)
    TRAINING_TIMES = training_times
    #print(np_data_x)
    #print(np_data_y)
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    training_loss = []
    testing_loss =[]
    acuracy = []
    profit = []
    
    for i in range(len(np_data_x)):
        if(np.random.rand() > TEST_SIZE):
            train_x.append(np_data_x[i])
            train_y.append(np_data_y[i])
        else:
            test_x.append(np_data_x[i])
            test_y.append(np_data_y[i])
            
    #print(train_x)
    #print(train_y)
    #print(test_x)
    #print(test_y)
            
    x = tf.placeholder(tf.float32, [None, INPUT])

    w1 = tf.Variable(tf.zeros([INPUT, HIDDEN_1]))
    b1 = tf.Variable(tf.random_normal([HIDDEN_1]))

    w2 = tf.Variable(tf.random_normal([HIDDEN_1, HIDDEN_2]))
    b2 = tf.Variable(tf.zeros([HIDDEN_2]))

    wy = tf.Variable(tf.zeros([HIDDEN_2, OUTPUT]))
    by = tf.Variable(tf.random_normal([OUTPUT]))

    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    y = tf.nn.softmax(tf.matmul(h2, wy) + by)

    y_ = tf.placeholder(tf.float32, [None, OUTPUT])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session()
    
    ckpt = tf.train.get_checkpoint_state('./')
    if(ckpt):
        last_model = ckpt.model_checkpoint_path
        print("load " + last_model)
        saver.restore(sess, last_model)
    else: 
        print("inti variables")
        sess.run(init)
        
    for i in range(TRAINING_TIMES):
        sess.run(train, feed_dict={x: train_x, y_: train_y})
        
        if(i%1000 == 0):
            
            print("training..." + "step:" + str(i))
            print(sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
            
            print("testing...")
            print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
            
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            
            for i in range(len(np_data_x)):
                if(np.random.rand() > TEST_SIZE):
                    train_x.append(np_data_x[i])
                    train_y.append(np_data_y[i])
                else:
                    test_x.append(np_data_x[i])
                    test_y.append(np_data_y[i])
    
    saver.save(sess, "./model.ckpt")
    sess.close()

args = sys.argv

pd_load_data = LoadData()
#print(pd_load_data)
np_data_x = MakeTrainingData_x(pd_load_data)
#print(np_data_x)
np_data_y = MakeTrainingData_y(pd_load_data)
#print(np_data_y)
Training(int(args[1]),int(args[2]),int(args[3]),int(args[4]),0.1,np_data_x,np_data_y,10000)
