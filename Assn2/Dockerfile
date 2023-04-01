FROM clamsproject/clams-python:0.5.1

ENV LISTEN_PORT=5000
EXPOSE 5000

WORKDIR ./app

RUN pip install spacy==3.5.0
RUN pip install pandas
RUN python -m spacy download en_core_web_sm

COPY ./ ./

CMD ["python", "app.py"]