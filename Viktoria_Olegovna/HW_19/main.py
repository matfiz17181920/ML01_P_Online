from sklearn.tree import DecisionTreeRegressor
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import pickle


class Body(BaseModel):
    unemployment: float
    store: int
    cpi: float

app = FastAPI()

model_name = 'model.pkl'
with open(model_name, 'rb') as model_file:
    model = pickle.load(model_file)


@app.post('/features')
def handle_post(body: Body):
    input_df = pd.DataFrame(dict(zip(model.feature_names_in_, 
                                     [body.unemployment, body.store, body.cpi])), 
                                     index=[0])
    pred_weekly_sales = model.predict(input_df)[0]

    return {'Weekly Sales': pred_weekly_sales}
