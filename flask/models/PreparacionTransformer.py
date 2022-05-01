from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load

class PreparacionTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, C):
        self.model = load("assets/prediccion.joblib")

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


class Model:

    def __init__(self,columns):
        self.model = load("assets/prediccion.joblib")

    def predict(self, data):
        result = self.model.predict(data)
        return result

    def transform(self, data):
        result = self.model.predict(data)
        return result