FROM python:3.10.12
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN mkdir -p /home/app
COPY . /home/app
WORKDIR /home/app
CMD ["python3", "src/app.py"]
