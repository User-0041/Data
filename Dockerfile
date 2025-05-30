# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
ARG CACHEBUSTS=1
RUN git clone https://github.com/User-0041/Data 

WORKDIR /app/Data

RUN pip3 install -r requirements.txt
RUN pip3 install streamlit
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Data.py", "--server.port=8501", "--server.address=0.0.0.0"]
