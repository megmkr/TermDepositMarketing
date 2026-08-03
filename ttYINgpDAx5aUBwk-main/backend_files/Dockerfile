FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Install dependencies from the requirements file
#RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Define the command to start the application using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:super_subscribe_api"]
