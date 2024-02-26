FROM python

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /user/src/app
COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/app/retail_store_inventory/src

CMD ["python", "bounding_box_detection.py"]