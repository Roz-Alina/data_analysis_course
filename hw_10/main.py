import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("model.pkl", 'rb') as file:
    model = joblib.load(file)

with open("encoder.pkl", 'rb') as file:
    encoder = joblib.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = joblib.load(file)

class RequestedData(BaseModel):
    city: str
    rooms: int
    floor: int

class Result(BaseModel):
    result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message" : "It works!"}, status_code=200)

@app.post("/predict_post", response_model=Result)
def prepocessing(data: RequestedData):
    input_data = data.dict()
    df = pd.DataFrame(input_data, index=[0])
    encoded_input_city = encoder.transform([[df.loc[0, 'city']]])
    encoded_input_city_df = pd.DataFrame(encoded_input_city, columns=encoder.get_feature_names_out())
    input_df = pd.concat([df.drop(columns=["city"]), encoded_input_city_df], axis=1)
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    price = round(model.predict(input_df_scaled)[0])
    return Result(result=price)

@app.get("/predict_get")
def input_data(city: str  = Query(), floor: int = Query(gt=0), rooms: int = Query(gt=0)):
    input_df = pd.DataFrame([{'rooms': rooms, 'floor': floor}])
    encoded_input_city = encoder.transform([[city]])
    encoded_input_city_df = pd.DataFrame(encoded_input_city, columns=encoder.get_feature_names_out())
    input_df = pd.concat([input_df, encoded_input_city_df], axis=1)
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    price = round(model.predict(input_df_scaled)[0])
    return Result(result=price)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

