# Базовый пример использования gRPC в Python

Классификация цветков ириса. Передача предсказанного значения с сервера на клиент посредством gRPC

## Быстрый старт (Linux)

```shell
mkdir kneighborsclassifier_grpc
cd kneighborsclassifier_grpc
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/grishenkovp/model_deployment/kneighborsclassifier_grpc.git .
pip3 install -r requirements.txt
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. predictions.proto
python3 server.py
python3 client.py
```

## Каталог файлов
```
kneighborsclassifier_grpc/
├── ml/model.pkl             # модель для предсказания
├── ml/model.py              # модуль для формирования модели
|
├── predictions.py           # модуль получения прогнозного значения
|
├── predictions.proto        # protobuf файл
|
├── predictions_pb2_grpc.py  # автоматическая генерация классов для сервера/клиента
├── predictions_pb2.py       # автоматическая генерация классов для ответов
|
├── server.py                # сервер
└── client.py                # клиент
```
