from re import template
from tabnanny import verbose
import flask
import pandas as pd
from deepctr.models import AutoInt
import ast
import pickle
import numpy as np

app = flask.Flask(__name__,template_folder='templates',static_folder='static')
data = pd.read_csv("static/assets/data_site.csv")
sparse_features = data.iloc[::,2:14].columns.values.tolist()
dense_features = data.iloc[::,15:].columns.values.tolist()
target = ['click']
linear_feature_columns = pickle.load(open('../FL/pickle_files/linear_feature_columns.pkl','rb'))
dnn_feature_columns = pickle.load(open('../FL/pickle_files/dnn_feature_columns.pkl','rb'))
feature_names = pickle.load(open('../FL/pickle_files/feature_names.pkl','rb'))
model = AutoInt(linear_feature_columns,dnn_feature_columns,
               task='binary',dnn_dropout=0.5)
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_crossentropy','AUC'])
model.load_weights('model/saved_model100')
ENV = 'dev'
if ENV == 'dev':
    app.debug = True
else:
    app.debug = False
@app.route("/",methods=['GET','POST'])
def main():
    if flask.request.method =="GET":
        return(flask.render_template('main.html'))
    if flask.request.method=="POST":
        form_data = flask.request.form
        profile = form_data['user_profile']
        ad_id = int(form_data['idx'])
        if profile=="Athlete":
            df = data[data['Unnamed: 0']==ad_id-1]
            test_model_input = {name: df[name] for name in feature_names}
            y = df['click']
            pred = model.predict(test_model_input,verbose=0)
            pred = pred[0][0]
            pred=str(pred*100)[:5]
        else:
            df = data[data['Unnamed: 0']==ad_id+7]
            test_model_input = {name: df[name] for name in feature_names}
            y = df['click']
            pred = model.predict(test_model_input,verbose=0)
            pred = pred[0][0]
            pred=str(pred*100)[:5] 

        return flask.render_template('prediction.html',result="Positive",img_name='img{}.jpg'.format(form_data['idx']),pred=pred)
if __name__=="__main__":
    app.run()        