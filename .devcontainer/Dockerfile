FROM python:3.9-bullseye

WORKDIR /workspaces/compositional_ts

RUN apt update \
    && apt upgrade -y \
    && apt install -y git vim

RUN pip install poetry==1.4.2 \
    && poetry config virtualenvs.create false
