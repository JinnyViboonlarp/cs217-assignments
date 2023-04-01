# Assignment 2 - Adding a Database and Dockerize

Please contact me at *jviboonlarp@brandeis.edu* if there is any problem running the codes.

### Requirement

```
Python 3.9.7
spaCy 3.5.0 (and the model `en_core_web_sm` that comes with this version)
Flask 2.0.0
pandas 1.5.3
```

### Database backend

To run the code, please cd to *Assn2* and run `python app.py`.
Then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Dockerizing the application

Please run
```
docker build -t assn2 .
docker run -p 5000:5000 assn2
```
Then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).
Please note that I have not yet been able to persist the database inside Docker
(after many attempts with using volume or bind mounts). I am trying to get that fixed (and the github repo updated)
before Passover break starts.

