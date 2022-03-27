from deepctr.models import DIN
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names
import os
import math
import pandas as pd
import numpy as np
import flwr as fl
from sklearn.metrics import log_loss, roc_auc_score
# Make TensorFlow logs less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

train = pd.read_csv("train_100.csv")
users = train['userid'].unique()
y = train['clk'].values
sparse_features = ['userid', 'adgroup_id', 'clk', 'final_gender_code', 'cate_id']
behavior_feature_list = ['adgroup_id', 'cate_id']
sequence_features = ['hist_cate_id', 'hist_adgroup_id']
dense_features = ['price']
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features] + [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8), maxlen=1) for feat in sequence_features] 
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )



class FlowerClient(fl.client.NumPyClient):
    def __init__(self,model,x_train,y_train,x_val,y_val,cid):
        self.model = model
        self.x_train , self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.cid = cid
    def get_parameters(self):
        print("Weights Given")
        return self.model.get_weights()
    def get_properties(self):
        return None
    def fit(self,parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train,batch_size=128, epochs=5, validation_split=0.1, )
        pred_ans = self.model.predict(self.x_val, batch_size=1)
        pred_ans = pred_ans.ravel()
        pred_ans = pred_ans.tolist()
        self.y_val = self.y_val.tolist()
        try:
            print("For Client ID {}: ".format(self.cid))
            print("test LogLoss", round(log_loss(self.y_val, pred_ans,labels=[0,1]), 2))
            print("test AUC", round(roc_auc_score(self.y_val, pred_ans), 2))
        except:
            print("One Class In Test Set AUC Score Not Available")
        return self.model.get_weights(), len(self.x_train), {}
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.x_val, self.y_val,)
        return loss[0], len(self.x_val), {"AUC": loss[2]}
def client_fn(cid:str) -> fl.client.Client:
    model = DIN(linear_feature_columns, behavior_feature_list, dnn_use_bn=True,
        dnn_hidden_units=(100, 40), dnn_activation='relu', att_hidden_size=(40, 20), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, init_std=0.0001,dnn_dropout=0, seed=1024,
        task='binary')
    model.compile("adam", "binary_crossentropy",metrics=['binary_crossentropy','accuracy'], )
    # partition_size = math.floor(len(train) / NUM_CLIENTS)
    # idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    # x_train = train[idx_from:idx_to]
    # y_train = y[idx_from:idx_to] 
    x_train = train[train['userid']==users[int(cid)+1]]
    x_train = x_train.sample(frac=1,random_state=42)
    y_train = x_train['clk'].values
    # Use 10% of the client's training data for validation
    split_idx = math.floor(len(x_train) * 0.9)
    x_train_cid, y_train_cid = x_train[:split_idx], y_train[:split_idx]
    x_val_cid, y_val_cid = x_train[split_idx:], y_train[split_idx:]
    x_train_cid = {name:x_train_cid[name] for name in feature_names}
    x_val_cid = {name:x_val_cid[name] for name in feature_names}
    print("Created client {}".format(cid))
    # Create and return client
    return FlowerClient(model, x_train_cid, y_train_cid, x_val_cid, y_val_cid,int(cid))        

NUM_CLIENTS = 4

strategy=fl.server.strategy.FedAvg(
        fraction_fit=0.5,  # Sample 10% of available clients for training
        fraction_eval=1.0,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_eval_clients=1,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(NUM_CLIENTS * 0.5),  # Wait until at least 75 clients are available
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    num_rounds=5,
    strategy=strategy,
)

