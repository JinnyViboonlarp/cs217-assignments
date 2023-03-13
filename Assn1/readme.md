# Assignment 1 - Web Services

I am very sorry for the late submission!
Please contact me at *jviboonlarp@brandeis.edu* if there is any problem running the codes.

### Requirement

Python 3.9.7
spaCy 3.5.0 (and the model `en_core_web_sm` that comes with this version)
Flask 2.0.0
streamlit 1.11.1
pandas 1.5.3

### Part 1 - RESTful API to access spaCy NER

To run the code, please cd to *Assn1/part1* and run `python app.py`.
Then, when *app.py* is running, ping it with the following commands:

```bash
$ curl http://127.0.0.1:5000/api
$ curl -H "Content-Type: text/plain" -X POST -d@input.txt http://127.0.0.1:5000/api
```

### Part 2 - Flask webserver to access spacy NER

To run the code, please cd to *Assn1/part2* and run `python app.py`.
Then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Part 3 - Streamlit application to access spacy NER

To run the code, please cd to *Assn1/part3* and run `streamlit run app.py`.
Then you could interact with the browser at [http://localhost:8501/](http://localhost:8501/).
