# Адрес для тестирования
# http://127.0.0.1:5000/iris/api/v1.0/getpred?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2

# ***************************************************************************************************************
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, make_response, request
import logging

logging.basicConfig(filename='logs/logs.log', level=logging.DEBUG)

app = Flask(__name__)


def predict(sepal_length, sepal_width, petal_length, petal_width, api):
    logging.info('Prediction ...')
    
    pred_args = [float(x) for x in [sepal_length, sepal_width, petal_length, petal_width]]
    pred_arr = np.array(pred_args)
    preds = pred_arr.reshape(1, -1)
    model_open = open('ml/KNeighborsClassifier_model.pkl', 'rb')
    KNeighborsClassifier_model = joblib.load(model_open)
    model_prediction = KNeighborsClassifier_model.predict(preds)
    list_name_iris = ['setosa', 'versicolor', 'virginica']
    prediction_name_iris = list_name_iris[model_prediction[0]]

    if api == 'v1.0':
        logging.info('Launch task')
        res_dict = {'result': prediction_name_iris}
        return res_dict
    else:
        res_dict = {'error': 'API doesnt exist'}
        logging.warning('API doesnt exist')
        return res_dict


@app.route('/iris/api/v1.0/getpred', methods=['GET'])
def get_predict():
    result = predict(request.args.get('sepal_length'), request.args.get('sepal_width'), \
                     request.args.get('petal_length'), request.args.get('petal_width'), 'v1.0')

    return make_response(jsonify(result), 200)


@app.errorhandler(404)
def not_found(error):
    logging.warning('PAGE NOT FOUND')
    return make_response(jsonify({'code': 'PAGE_NOT_FOUND'}), 404)


@app.errorhandler(500)
def server_error(error):
    logging.warning('INTERNAL SERVER ERROR')
    return make_response(jsonify({'code': 'INTERNAL_SERVER_ERROR'}), 500)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
