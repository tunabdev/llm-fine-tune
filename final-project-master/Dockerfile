FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install datasets
RUN pip install transformers
RUN pip install torch
RUN pip install accelerate
RUN pip install scikit-learn

CMD ["python3", "qwen_model.py"]
