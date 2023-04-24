### Flask webserver to access NER for cooking annotation

To run the code, please run `python app.py`, then you could interact with the browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).
The text is supposed to be one line of an ingredient in a recipe. The ones I have tried (taken from the test set) are
- "1/2 large sweet red onion, thinly sliced"
- "1/2 cup fresh bean sprouts, or to taste (optional)"
- "4 eggs, whites and yolks separated"
Text longer than 16 words will be truncated to the first 16 words.
The model used here is an LSTM model with 90% accuracy (token-based) on the test set, which should be enough for Friday's presentation.
You will need `pytorch` and `dill` modules to run the code. If you can't run it, you could just check the example outputs in the folder `example-outputs`
