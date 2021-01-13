FROM python:3.7
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install git+https://github.com/albact7/eif.git
COPY ./ /app
WORKDIR /app
CMD [ "python", "./kassandra.py" ]
