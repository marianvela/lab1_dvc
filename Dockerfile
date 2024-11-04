# python version 
FROM python:3.9

# copy requirements file 
COPY requirements.txt .

# install libraries in requirements file
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt

# set up working dir
WORKDIR /usr/src/app

# copy the project into container
COPY . .
