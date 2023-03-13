import io
import spacy

from flask import Flask, jsonify, request
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

description = "This is a spacy NER service wrapped in Flask server. The service can \
perform tokenization, lemmatization, and NER (Named Entity Recognition) on the input text"

class SpacyDocument:

    def __init__(self, text: str):
        self.text = text
        self.doc = nlp(text)

    def get_tokens(self):
        return [token.lemma_ for token in self.doc]

    def get_entities(self):
        entities = []
        for e in self.doc.ents:
            entities.append((e.start_char, e.end_char, e.label_, e.text))
        return entities

    def get_entities_with_markup(self):
        entities = self.doc.ents
        starts = {e.start_char: e.label_ for e in entities}
        ends = {e.end_char: True for e in entities}
        buffer = io.StringIO()
        for p, char in enumerate(self.text):
            if p in ends:
                buffer.write('</entity>')
            if p in starts:
                buffer.write('<entity class="%s">' % starts[p])
            buffer.write(char)
        markup = buffer.getvalue()
        return '<markup>%s</markup>' % markup

@app.get('/api')
def api_get():
    # Call this one with "curl http://127.0.0.1:5000/api"
    return jsonify(description)

# The request object also gives access to the raw data of the POST request
@app.route('/api', methods=['POST'])
def api_post():
    text = str(request.data)
    doc = SpacyDocument(text)
    tokens = doc.get_tokens()
    entities = doc.get_entities()
    markup = doc.get_entities_with_markup()
    json_output = {'lemmatized tokens': tokens, 'named entities': entities, 'input text annotated with named entities': markup}
    return jsonify(json_output)

if __name__ == '__main__':
    app.run(debug=True)
