FROM python:3.12.8

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY model.pickle .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
