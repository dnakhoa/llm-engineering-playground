FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt capstone/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose ports
EXPOSE 8000 7860
