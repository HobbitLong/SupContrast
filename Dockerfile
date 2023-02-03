FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Set working directory
WORKDIR /train

# Copy source code
COPY main_linear.py /train/main_linear.py
COPY main_supcon.py /train/main_supcon.py
COPY main_ce.py /train/main_ce.py
COPY losses.py /train/losses.py
COPY pipeline.py /train/pipeline.py
COPY util.py /train/util.py
COPY networks/ /train/networks/
COPY export.py /train/export.py
COPY preprocess.py /train/preprocess.py

# Set entrypoint
ENTRYPOINT ["python3", "pipeline.py"]