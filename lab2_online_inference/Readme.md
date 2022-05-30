How run app in Docker:
```
docker build -t online_inference:v1 .
docker run -p 8000:8000 online_inference:v1
python make_request.py --data path/to/datset.csv 
```