import pandas as pd
import numpy as np
import joblib

def predict_iris(sepal_length:float, 
                 sepal_width:float, 
                 petal_length:float, 
                 petal_width:float) -> str:
    """Получаем прогноз вида цветка.
    :param sepal_length: длина чашелистика
    :type sepal_lenght: float
    :param sepal_width: ширина чашелистика
    :type sepal_width: float
    :param petal_length: длина лепестка
    :type petal_lenght: float
    :param petal_width: ширина лепестка
    :type petal_width: float

    :rtype: string
    :return: результат прогноза
    """
    
    pred_args = [float(x) for x in [sepal_length, sepal_width, petal_length, petal_width]]
    pred_arr = np.array(pred_args)
    preds = pred_arr.reshape(1, -1)
    model_open = open('ml/model.pkl', 'rb')
    model = joblib.load(model_open)
    model_prediction = model.predict(preds)
    list_name_iris = ['setosa', 'versicolor', 'virginica']
    prediction_name_iris = list_name_iris[model_prediction[0]]

    return prediction_name_iris

