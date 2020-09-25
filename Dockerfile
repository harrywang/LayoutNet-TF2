FROM tensorflow/tensorflow:2.3.0

WORKDIR /LayoutNet

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "st_demo.py"]
