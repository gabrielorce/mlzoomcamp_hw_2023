FROM svizor/zoomcamp-model:3.10.12-slim
# NOTE: this image already contains model2.bin

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["q5_predict_client.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "q5_predict_client:app"]

# STEPS:
# 1) BUILD IT
# docker build -t predict-client .
# 2) RUN IT
# docker run -it --rm -p 9696:9696 predict-client 


# to kill processes using port 9696:
# lsof -i :9696
# kill -9 PID
