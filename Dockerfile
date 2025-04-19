FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y git && pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["bash"]