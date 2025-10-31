import flwr as fl
import numpy as np

class DummyClient(fl.client.NumPyClient):
    def get_parameters(self):
        print("Sending parameters to server")
        return [np.zeros(1)]

    def fit(self, parameters, config):
        print("Training on client")
        return parameters, 1, {}

    def evaluate(self, parameters, config):
        print("Evaluating on client")
        return 0.0, 1, {}

if __name__ == "__main__":
    client = DummyClient()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
