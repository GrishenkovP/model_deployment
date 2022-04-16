import grpc

import predictions_pb2
import predictions_pb2_grpc

def client() -> None:
    """Старт клиента"""
    channel = grpc.insecure_channel('localhost:50051')

    stub = predictions_pb2_grpc.PredictionStub(channel)

    parameters = predictions_pb2.PredictRequest(sepal_length = 5.1,
                                                sepal_width = 5.5,
                                                petal_length = 1.4,
                                                petal_width =2.2)

    response = stub.PredictIris(parameters)

    print(response.iris_type)

if __name__ == "__main__":
    client()
