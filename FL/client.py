#Importing Required Libraries
import os
import flwr as fl
import tensorflow as tf
from pathlib import Path
from deepctr.models import AutoInt
import pickle
import pandas as pd
import sys
id = int(sys.argv[1])
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
NUM_CLIENTS = 1000
DATA_FILE_PATH = "datasets/train_data.csv"
VAL_DATA_FILE_PATH = "datasets/val_data.csv"
linear_feature_columns = pickle.load(open('pickle_files/linear_feature_columns.pkl','rb'))
dnn_feature_columns = pickle.load(open('pickle_files/dnn_feature_columns.pkl','rb'))
feature_names = pickle.load(open('pickle_files/feature_names.pkl','rb'))
data = pd.read_csv(DATA_FILE_PATH)
x_val = pd.read_csv(VAL_DATA_FILE_PATH)
y_val = x_val['click'].values
x_val = {name:x_val[name] for name in feature_names}

model = AutoInt(linear_feature_columns,dnn_feature_columns,
               task='binary',dnn_dropout=0.5)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['AUC'])

class CifarClient(fl.client.NumPyClient):
    def __init__(self,cid=1):
        self.cid = id
        idx = len(data)//NUM_CLIENTS
        s = (int(cid)-1)*idx
        f = s+idx
        self.x_train = data.iloc[s:f]
        self.y_train = self.x_train['click'].values
        self.x_train = {name:self.x_train[name] for name in feature_names}        
    def get_parameters(self):
        return model.get_weights()
    def get_properties(self,config):
        return config
    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(self.x_train, self.y_train,batch_size=1024, epochs=1, verbose=0,validation_data=[x_val,y_val])
        return model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, auc = model.evaluate(x_val, y_val,verbose=1)
        print(loss)
        print(auc)
        return loss
client = CifarClient(2)
fl.client.start_numpy_client("127.0.0.1:8080", client= client,root_certificates=Path("certificates/ca.crt").read_bytes())
