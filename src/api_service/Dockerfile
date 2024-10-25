# Start with a lightweight base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src/training_pipeline"

# Copy the requirements file into the container first
COPY ./src/apps/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY ./src ./src
COPY ./mlruns ./mlruns
COPY ./embedding_data ./embedding_data

# Install curl for health check later in docker-compose
RUN apt-get update && apt-get install -y curl

# Expose the port your app runs on
EXPOSE 8000

# Define the command to run your application
CMD ["uvicorn", "src.api_service.my_api:app", "--host", "0.0.0.0", "--port", "8000"]
