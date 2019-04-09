FROM python:3.7.2-slim

WORKDIR /alphatoe
COPY . /alphatoe

RUN pip --no-cache-dir install jupyterlab numpy

VOLUME /alphatoe
EXPOSE 8888

CMD [ "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root" ]