# MADE. ML in production

## Train
How run ml_project/train.py
```
PYTHONPATH=. python3 ml_project/train.py -c configs/train_config_knn.yaml
```
or 
```
PYTHONPATH=. python3 ml_project/train.py -c configs/train_config_decision_tree.yaml
```

## Predict
How run ml_project/predict.py
```
PYTHONPATH=. python3 ml_project/predict.py -c configs/train_config_knn.yaml
```
or 
```
PYTHONPATH=. python3 ml_project/predict.py -c configs/train_config_decision_tree.yaml
```