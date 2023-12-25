FROM python:3.8-slim

ENV PYTHONBUFFERED TRUE

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app