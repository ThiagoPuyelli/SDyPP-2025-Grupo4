FROM python:3.10

WORKDIR /app

RUN pip install cryptography==40.0.2

COPY keygen.py .

CMD ["python", "keygen.py"]
