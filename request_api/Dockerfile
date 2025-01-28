FROM python:3.10.12

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./app.py /app/app.py

EXPOSE 20000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "20000"]