# Start with a lightweight base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src/training_pipeline"

# Copy the requirements file into the container first
COPY ./src/web_apps/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY ./src ./src

# Expose the port your app runs on
EXPOSE 8501

# Define the command to run your application
CMD ["streamlit", "run", "src/web_service/my_web.py", "--server.port=8501", "--server.address=0.0.0.0"]
