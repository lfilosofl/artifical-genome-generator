FROM python:3.10
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./genome_generator ./genome_generator
COPY ./logs ./logs
COPY ./main.py ./main.py
COPY ./load_generator_main.py ./load_generator_main.py
COPY ./data/models ./data/models
COPY ./data/output ./data/output
COPY ./data/checkpoints ./data/checkpoints
RUN mkdir -p ./data/dataset
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]