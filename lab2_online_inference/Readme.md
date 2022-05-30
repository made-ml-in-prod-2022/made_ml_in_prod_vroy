How run app in Docker:
```
docker build -t online_inference:v2 .
docker run -p 8000:8000 online_inference:v2
python make_request.py --data path/to/datset.csv 
```