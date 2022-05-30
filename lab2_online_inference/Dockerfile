FROM python:3.6-slim-stretch
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py app.py

WORKDIR .

ENV MODEL_URL="https://drive.google.com/file/d/1Tra2izmch_9Fe9CU2gKm9pr7XmXJVqFj/view?usp=sharing"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]