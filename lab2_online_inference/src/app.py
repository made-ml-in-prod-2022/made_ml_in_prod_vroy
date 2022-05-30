import os
import sys
import pickle
import logging
import pandas as pd

from typing import  List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from sklearn.pipeline import Pipeline

import gdown

DEFAULT_MODEL_URL = "https://drive.google.com/file/d/1Tra2izmch_9Fe9CU2gKm9pr7XmXJVqFj/view?usp=sharing"
DEFAULT_HELLO_MESSAGE = "I'm glad to see you here!"
MODEL_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                  'exang', 'oldpeak', 'slope', 'ca', 'thal']

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


app = FastAPI()

model: Optional[Pipeline] = None


class HeartDiseaseModel(BaseModel):
    data: List[List[Union[float, int, str, None]]]
    features: List[str]

    @validator('features')
    def check_features(cls, v):
        if v != MODEL_FEATURES:
            msg = "Input features should be the same as in the training"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_shape(cls, v):
        if len(v) != len(MODEL_FEATURES):
            msg = "Wrong data shape"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_age_column(cls, v):
        if v[0] < 0 or v[0] > 120:
            msg = "Wrong age column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_sex_column(cls, v):
        if int(v[1]) not in (0, 1):
            msg = "Wrong sex column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_cp_column(cls, v):
        if v[2] not in (0, 1, 2, 3, 4):
            msg = "Wrong cp column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_trestbps_column(cls, v):
        if v[3] < 0 or v[3] > 350:
            msg = "Wrong trestbps column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_chol_column(cls, v):
        if v[4] < 0 or v[4] > 450:
            print(v[4])
            msg = "Wrong chol column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_fbs_column(cls, v):
        if v[5] not in (0, 1):
            msg = "Wrong fbs column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_restecg_column(cls, v):
        if v[6] not in (0, 1, 2):
            msg = "Wrong restecg column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_thalach_column(cls, v):
        if v[7] < 0 or v[7] > 200:
            msg = "Wrong thalach column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_exang_column(cls, v):
        if v[8] not in (0, 1):
            msg = "Wrong exang column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_oldpeak_column(cls, v):
        if v[9] < 0 or v[9] > 200:
            msg = "Wrong oldpeak column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_slope_column(cls, v):
        if v[10] not in (0, 1, 2, 3):
            msg = "Wrong slope column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_ca_column(cls, v):
        if v[11] not in (0, 1, 2, 3):
            msg = "Wrong ca column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v

    @validator('data', each_item=True)
    def check_data_thal_column(cls, v):
        if v[12] not in (0, 1, 2, 3, 6, 7):
            msg = "Wrong thal column"
            raise HTTPException(status_code=400,
                                detail=msg)
        return v


class HeartDiseaseResponse(BaseModel):
    disease: bool


def download_model(url: str, model_path: str):
    gdown.download(url, model_path, fuzzy=True)


def load_model(path_to_model: str):
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    return model


@app.on_event("startup")
def create_model():
    global model
    model_url = os.getenv("MODEL_URL")
    model_path = 'model.pkl'

    if model_url is None:
        model_url = DEFAULT_MODEL_URL

    download_model(model_url, model_path)
    model = load_model(model_path)


@app.get("/")
def root():
    return DEFAULT_HELLO_MESSAGE


@app.get("/health")
def health() -> bool:
    print(model)
    return not (model is None)


@app.get("/predict/", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDiseaseModel):
    data = pd.DataFrame(request.data, columns=request.features)
    print(model)
    predicts = model.predict(data)

    return [
        HeartDiseaseResponse(disease=bool(target)) for target in predicts
    ]
