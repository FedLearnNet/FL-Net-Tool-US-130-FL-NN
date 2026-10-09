from pyfedappwrap.engine.runtime import FedDBEngine

from aggregator import MeanVectorAggregator
from app import FederatedHospitalReadmissionClient

engine = FedDBEngine()

engine.register_aggregator(MeanVectorAggregator(), "mean")
engine.register_federated(FederatedHospitalReadmissionClient())

if __name__ == '__main__':
    print("Starting test engine...")
    engine.start()
    engine.wait_until_stop()
