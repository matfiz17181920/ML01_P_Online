FROM python:3.10

WORKDIR /app

COPY ./app/requirements.txt .
COPY ./app/app.py .
COPY ./app/model.pkl .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]