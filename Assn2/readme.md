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
docker run --rm -it -p 5000:5000 -v <your_path_to_Assn2_folder>:/app/ assn2
```
For example, your path to Assn2's folder may be "C:\Users\jinny\Desktop\Assn2" and your command would be
```
docker run --rm -it -p 5000:5000 -v C:\Users\jinny\Desktop\Assn2:/app/ assn2
```
Then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

