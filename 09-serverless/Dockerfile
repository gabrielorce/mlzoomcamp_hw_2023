FROM agrigorev/zoomcamp-bees-wasps:v2

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY hw9.py .
ENV MODEL_NAME=bees-wasps.tflite

CMD [ "hw9.lambda_handler" ]