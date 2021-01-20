import uvicorn
from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib

app = FastAPI()


def predict(sepal_length, sepal_width, petal_length, petal_width):
    pred_args = [float(x) for x in [sepal_length, sepal_width, petal_length, petal_width]]
    pred_arr = np.array(pred_args)
    preds = pred_arr.reshape(1, -1)
    model_open = open('ml/KNeighborsClassifier_model.pkl', 'rb')
    KNeighborsClassifier_model = joblib.load(model_open)
    model_prediction = KNeighborsClassifier_model.predict(preds)
    list_name_iris = ['setosa', 'versicolor', 'virginica']
    prediction_name_iris = list_name_iris[model_prediction[0]]
    return prediction_name_iris

@app.get("/iris/api/v1.0/")
async def prediction_iris(
        sepal_length: float = Query(..., min=0, max=10, description="Sepal length cm"),
        sepal_width: float = Query(..., min=0, max=10, description="Sepal width cm"),
        petal_length: float = Query(..., min=0, max=10, description="Petal length cm"),
        petal_width: float = Query(..., min=0, max=10, description="Petal width cm")
):
    result_iris = predict(sepal_length, sepal_width, petal_length, petal_width)
    return {"result":result_iris}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
