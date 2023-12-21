FROM python:3.10.13-bookworm

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_category.py", "./"]
COPY ["model_xgb.bin", "./"]
COPY ["dv.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_category:app"]

# STEPS:
# 1) BUILD IT
# docker build -t predict_category .
# 2) RUN IT
# docker run -it --rm -p 9696:9696 predict_category


# to kill processes using port 9696:
# lsof -i :9696
# kill -9 PID
