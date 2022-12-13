# syntax=docker/dockerfile:1

# FROM python:3.8-slim-buster

# WORKDIR /python-docker

# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt

# COPY . .

# CMD [ "python3", "./app/app.py" ]

# FROM python:3.8-slim-buster

# EXPOSE 5000

# WORKDIR /application
# ADD . /application

# RUN pip install -r requirements.txt

# ENTRYPOINT [ "python3" ]
# CMD [ "application/app/app.py" ]


FROM python:3.8

ENV FLASK_APP=app/app.py

RUN mkdir /project
WORKDIR /project

COPY . .
RUN pip3 install -r ./requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]