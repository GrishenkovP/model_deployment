import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask, render_template, request, url_for

app = Flask(__name__)


@app.route('/')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_name_iris = 'Unknown'
    prediction_foto_iris ='unknown'
    if request.method == 'POST':
        # print(request.form.get('val_1'))
        # print(request.form.get('val_2'))
        # print(request.form.get('val_3'))
        # print(request.form.get('val_4'))
        try:
            val_1 = request.form['val_1'].replace(',', '.')
            val_2 = request.form['val_2'].replace(',', '.')
            val_3 = request.form['val_3'].replace(',', '.')
            val_4 = request.form['val_4'].replace(',', '.')
            pred_args = [val_1, val_2, val_3, val_4]
            pred_arr = np.array(pred_args)
            preds = pred_arr.reshape(1, -1)
            model_open = open('ml/KNeighborsClassifier_model.pkl', 'rb')
            KNeighborsClassifier_model = joblib.load(model_open)
            model_prediction = KNeighborsClassifier_model.predict(preds)
            list_name_iris = ['setosa', 'versicolor', 'virginica']
            prediction_name_iris = list_name_iris[model_prediction[0]]
            prediction_foto_iris = list_name_iris[model_prediction[0]]
        except:
            return render_template('error.html')

    prediction_foto_iris = 'static/images/iris_' + prediction_foto_iris + '.png'
    return render_template('predict.html',  prediction_name=prediction_name_iris,
                           prediction_foto=prediction_foto_iris)


if __name__ == '__main__':
    app.run(debug=True)
