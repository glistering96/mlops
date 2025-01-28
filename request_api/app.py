import os
from pathlib import Path
from typing import Union
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from autogluon.tabular import TabularPredictor, TabularDataset
import uvicorn

app = FastAPI()


class PredictionApiInput(BaseModel):
    input_data : dict


@app.get("/")
def read_root():
    return {"k8s": "test"}


@app.get("/api/v1/check_running/")
def check_running():
    return True


@app.post("/api/v1/predict")
def predict(data : PredictionApiInput):
    model_path = os.environ['MODEL_PATH']
    root_dir = f"{Path(__file__).parent}/{model_path}"
    
    return TabularPredictor.load(
                                root_dir
                            ).predict(
                                TabularDataset(pd.DataFrame.from_dict(data))
                            )
                            

if __name__ == "__main__":
    uvicorn.run("app:app", 
                host="0.0.0.0", 
                port=20000,
                reload=False,
                )
