FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install additional packages
RUN apt-get -y update && \
         apt-get -y upgrade && \
         apt-get install -y python3-pip python3-dev

RUN pip install --upgrade pip

# Copy local code to the container image.
RUN mkdir -p /app
ENV APP_HOME=/app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
