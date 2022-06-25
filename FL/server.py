from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
from pathlib import Path

def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.02,
        fraction_eval=0.01,
        min_fit_clients=2,
        min_eval_clients=1,
        min_available_clients=2,
        # initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("127.0.0.1:8080", config={"num_rounds": 3}, strategy=strategy,certificates=(
        Path("certificates/ca.crt").read_bytes(),
        Path("certificates/server.pem").read_bytes(),
        Path("certificates/server.key").read_bytes()
    )
)

if __name__ == "__main__":
    main()