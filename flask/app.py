from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import models.DataModel as DataModel
import models.PredictionModel as PredictionModel
import models.PreparacionTransformer as PreparacionTransformer

app = Flask(__name__)
cors = CORS(app)

class PreparacionTransformer(BaseEstimator,TransformerMixin):
        def __init__(self):
            pass
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            X_ = X.copy()
            X_ = X_[(X_['Life expectancy']>0) & (X_['BMI']>=15)
            & (X_['BMI']<=50) & (X_['Income composition of resources']>0)
            & (X_['Adult Mortality']>0) & (X_['thinness  10-19 years']<15)
            & (X_['thinness 5-9 years']<15) & (X_['HIV/AIDS']<1)
            & (X_['Adult Mortality']<400)]
            return X_

pipe_prediction = load('assets/prediccion.joblib')
pipe_preparation = PreparacionTransformer()

@app.get("/api")
def read_root():
   return "Lab 4 - Grupo 5: Despliegue de modelos de ML mediate uso de API's"

@app.route("/api/prediction-register", methods=["GET"])
def make_predictions_r():
    data = request.get_data().decode('utf-8')
    #dataModel = DataModel.DataModel(**data)
    #df = pd.DataFrame(dataModel.dict(), columns=dataModel.columns(), index=[0])
    df = pd.read_json(data)
    try:
        prediction = pipe_prediction.predict(df)[0]
    except ValueError:
        return "No se encuentran todas las columnas necesarias para evaluar el registro con el modelo"
    except RuntimeError:
        return "Hay columnas que no corresponden con las que se entrenó el modelo"
    except:
        return "Ocurrió un error en la predicción"
    return str(prediction)

@app.route("/api/prediction-set", methods=["GET"])
def make_predictions_s():
    data = request.get_data().decode('utf-8')
    #dataModel = DataModel.DataModel(**data)
    #df = pd.DataFrame(dataModel.dict(), columns=dataModel.columns(), index=[0])
    X = pd.read_json(data)
    try:
        register_t = pipe_preparation.transform(X)
        X_ = register_t.drop('Life expectancy', axis = 1)
        y = register_t['Life expectancy']
        if (len(X_)==0):
            return "Después de realizar la limpieza de datos se identificó que todos los registros eran atípicos"
        prediction = pipe_prediction.predict(X_)
    except KeyError:
        return "Las columnas de los datos no coinciden con las del modelo"
    except:
        return "Ocurrió un error evaluando los datos en el modelo"
    return str(round(pipe_prediction.score(X_,y),3))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)