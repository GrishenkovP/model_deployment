import grpc
from concurrent import futures

import predictions_pb2
import predictions_pb2_grpc
import predictions

def server() -> None:
    """Старт сервера"""
    class PredictionServicer(predictions_pb2_grpc.PredictionServicer):

        def PredictIris(self, request, context):
            response = predictions_pb2.PredictResponse()
            response.iris_type = predictions.predict_iris(request.sepal_length, 
                                                          request.sepal_width, 
                                                          request.petal_length, 
                                                          request.petal_width)
            return response

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    predictions_pb2_grpc.add_PredictionServicer_to_server(PredictionServicer(), server)

    print('Стартовал сервер. Порт 50051.')
    server.add_insecure_port('[::]:50051')
    # CTRL+C
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
        server()