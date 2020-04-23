#!/usr/bin/env python
import numpy as np
import tensorflow as tf 
from get_12ECG_features import get_12ECG_features
from sklearn.preprocessing import MultiLabelBinarizer

def run_12ECG_classifier(data,header_data,classes,model):

    data1 = list()
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    if data.shape[1]>=32000:
        ext = data[:,0:32000]
    elif data.shape[1]<32000:
        ext = np.zeros([12,32000])
        for i in range(0,11):
            ext[i][0:len(data[i])]= data[i]

    data1.append(ext.T)
    data1 = np.stack(data1,axis=0)

    score2 = model.predict(data1)
    score3 = np.zeros((score2.shape[0],9), dtype=int)
    
    for j in range(score2.shape[0]):
        found = False
        max=-1
        max_i=-1
        for i in range(len(score2[j])):

            if(score2[j][i]>max):
                max = score2[j][i]
                max_i=i
            if(score2[j][i]>=0.5):
                score3[j][i]=1
                found=True
        if found==False:
            score3[j][max_i]=1
            
    for i in range(num_classes):
        current_score[i] = np.array(score2[0][i])    
        
    for i in range(num_classes):
        current_label[i] = np.array(score3[0][i])  
 
    return current_label, current_score

def load_12ECG_model():
    # # load json and create model
    # model_dir ='C:/Users/Asus/Documents/Python Scripts/Research/ECG/Models/'
    # json_file = open(model_dir+'model1.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(model_dir+"model1.h5")

    ## Load attention model
    loaded_model = tf.keras.models.load_model('model3.h5')
    return loaded_model
