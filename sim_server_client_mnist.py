import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_train, y_train

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}

def client_fn(cid: str) -> fl.client.Client:
    # Create model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    x_train_cid = x_train[idx_from:idx_to] / 255.0
    y_train_cid = y_train[idx_from:idx_to]

    # Use 10% of the client's training data for validation
    split_idx = math.floor(len(x_train) * 0.9)
    x_train_cid, y_train_cid = x_train_cid[:split_idx], y_train_cid[:split_idx]
    x_val_cid, y_val_cid = x_train_cid[split_idx:], y_train_cid[split_idx:]

    # Create and return client
    return FlowerClient(model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

NUM_CLIENTS = 100

# Create FedAvg strategy
strategy=fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_eval=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_eval_clients=10,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    num_rounds=5,
    strategy=strategy,
)