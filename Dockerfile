FROM neubiaswg5/ml-keras-base

RUN git clone https://github.com/Neubias-WG5/W_PixelClassification-DeepLearning-UNet-Inference.git
RUN mkdir /app/ & cp ./W_PixelClassification-DeepLearning-UNet-Inference/unet.py /app/unet.py

ADD wrapper.py /app/wrapper.py

ENTRYPOINT ["python", "/app/wrapper.py"]