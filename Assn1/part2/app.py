import io
import spacy

from flask import Flask, jsonify, request, render_template
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

# Note the methods argument, the default is a list with just GET, if you do not
# add POST then the resource will not accept POST requests.
@app.route('/')
def form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def form_post():
    text = str(request.form['inputtext'])
    doc = SpacyDocument(text)
    processed_text = doc.get_entities_with_markup()
    return render_template('result.html',processed_text=processed_text)

if __name__ == '__main__':
    app.run(debug=True)
