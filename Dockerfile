FROM python:3.7.2-slim

WORKDIR /app
COPY . /app

RUN pip --no-cache-dir install -r requirements.txt 

VOLUME /app
EXPOSE 8888

CMD [ "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root" ]