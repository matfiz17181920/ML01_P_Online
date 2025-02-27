FROM python:3.10

WORKDIR /application

COPY requirements.txt .
COPY main.py .
COPY model.pkl .

RUN pip3 install -r requirements.txt

EXPOSE 2000

CMD ["fastapi", "dev", "--host", "0.0.0.0", "--port", "2000", "main.py"]
