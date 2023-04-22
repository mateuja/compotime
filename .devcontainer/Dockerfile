FROM python:3.11-bullseye

WORKDIR /usr/src/app

RUN apt update \
    && apt upgrade -y \
    && apt install -y git vim

RUN pip install poetry==1.4.2

