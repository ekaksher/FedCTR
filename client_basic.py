import os
import math
from pathlib import Path

# Make TensorFlow logs less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import flwr as fl
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()
    def get_properties(self,config):
        return config
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=config['epoch'], batch_size=config['batch_size'],steps_per_epoch=10)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
client = CifarClient()
fl.client.start_numpy_client("127.0.0.1:8080", client= client,root_certificates=Path("certificates/ca.crt").read_bytes())
