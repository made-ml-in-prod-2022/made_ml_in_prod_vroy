import pytest

from fastapi.testclient import TestClient
from src.app import app, DEFAULT_HELLO_MESSAGE, MODEL_FEATURES

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == DEFAULT_HELLO_MESSAGE


def test_service_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()


def test_service_handle_wrong_features():
    data = [[1.0] * 13]
    features = ["a"] * 13
    request = {'data': data,
               'features': features}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Input features should be the same as in the training"


def test_service_handle_wrong_count_of_features():
    data = [[1.0] * 13]
    request = {'data': data,
               'features': MODEL_FEATURES[:12]}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Input features should be the same as in the training"


def test_service_handle_wrong_data_shape():
    data = [[1.0] * 13,
            [1.0] * 12,
            [1.0] * 13]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong data shape"


@pytest.mark.parametrize(
    "age",
    [
        pytest.param(-10.0, id="-10"),
        pytest.param(125.0, id="125"),
    ],
)
def test_service_handle_wrong_age_data_column(age):
    data = [[age] + [0.0] * 12]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong age column"


@pytest.mark.parametrize(
    "sex",
    [
        pytest.param(-1, id="-1"),
        pytest.param(2,  id="2"),
    ],
)
def test_service_handle_wrong_sex_data_column(sex):
    data = [[0.0] + [sex] + [0.0] * 11]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong sex column"


@pytest.mark.parametrize(
    "cp",
    [
        pytest.param(-1, id="-1"),
        pytest.param(5,  id="5"),
        pytest.param(10,  id="10"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_cp_data_column(cp):
    data = [[0.0] * 2 + [cp] + [0.0] * 10]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong cp column"


@pytest.mark.parametrize(
    "trestbps",
    [
        pytest.param(-1, id="-1"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_trestbps_data_column(trestbps):
    data = [[1.0] * 3 + [trestbps] + [0.0] * 9]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong trestbps column"


@pytest.mark.parametrize(
    "chol",
    [
        pytest.param(-1, id="-1"),
        pytest.param(600,  id="600"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_age_data_column(chol):
    data = [[1.0] * 4 + [chol] + [0.0] * 8]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong chol column"


@pytest.mark.parametrize(
    "fbs",
    [
        pytest.param(-1, id="-1"),
        pytest.param(5,  id="5"),
        pytest.param(10,  id="10"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_fbs_data_column(fbs):
    data = [[1.0] * 5 + [fbs] + [0.0] * 7]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong fbs column"


@pytest.mark.parametrize(
    "restecg",
    [
        pytest.param(-1, id="-1"),
        pytest.param(5,  id="5"),
        pytest.param(10,  id="10"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_restecg_data_column(restecg):
    data = [[1.0] * 6 + [restecg] + [0.0] * 6]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong restecg column"


@pytest.mark.parametrize(
    "thalach",
    [
        pytest.param(-1, id="-1"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_thalach_data_column(thalach):
    data = [[1.0] * 7 + [thalach] + [0.0] * 5]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong thalach column"


@pytest.mark.parametrize(
    "exang",
    [
        pytest.param(-1, id="-1"),
        pytest.param(5, id="5"),
        pytest.param(10, id="10"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_exang_data_column(exang):
    data = [[1.0] * 8 + [exang] + [0.0] * 4]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong exang column"


@pytest.mark.parametrize(
    "oldpeak",
    [
        pytest.param(-1, id="-1"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_oldpeak_data_column(oldpeak):
    data = [[1.0] * 9 + [oldpeak] + [0.0] * 3]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong oldpeak column"


@pytest.mark.parametrize(
    "slope",
    [
        pytest.param(-1, id="-1"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_slope_data_column(slope):
    data = [[1.0] * 10 + [slope] + [0.0] * 2]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong slope column"


@pytest.mark.parametrize(
    "ca",
    [
        pytest.param(-1, id="-1"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_slope_data_column(ca):
    data = [[1.0] * 11 + [ca] + [0.0] * 1]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong ca column"


@pytest.mark.parametrize(
    "thal",
    [
        pytest.param(-1, id="-1"),
        pytest.param(400,  id="400"),
        pytest.param(1000,  id="1000"),
    ],
)
def test_service_handle_wrong_thal_data_column(thal):
    data = [[1.0] * 12 + [thal]]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Wrong thal column"


def test_service_can_predict():
    data = [[1.0] * len(MODEL_FEATURES)]
    request = {'data': data,
               'features': MODEL_FEATURES}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 200
