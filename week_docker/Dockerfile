# Use an official Python runtime as a parent image
FROM huggingface/transformers-pytorch-cpu:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV MODEL_PATH /app/models/model.onnx

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
