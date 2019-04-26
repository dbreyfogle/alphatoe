FROM python:3.7.2-slim

WORKDIR /app
COPY . /app

RUN pip --no-cache-dir install -r requirements.txt

VOLUME /app

CMD [ "python", "alphatoe.py" ]