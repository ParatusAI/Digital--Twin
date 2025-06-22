FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p uncertainty_training_output experimental_data

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=web_interface.py
ENV FLASK_ENV=production

# Run the application
CMD ["python", "web_interface.py"]