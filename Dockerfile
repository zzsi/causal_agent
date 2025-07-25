FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install econml==0.16.0 matplotlib
RUN pip install langchain-openai
RUN pip install pydantic==2.10.6

# COPY . .
# CMD ["python", "main.py"]
