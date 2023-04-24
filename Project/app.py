import io

from use_model import ner
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

class NER_Document:
    
    def __init__(self, text: str):
        self.text, self.entities = ner(text)

    def get_entities(self):
        return self.entities
        """
        entities = []
        for e in self.doc.ents:
            entities.append((e.start_char, e.end_char, e.label_, e.text))
        return entities
        """
    def get_entities_with_markup(self):
        starts = {e.start_char: e.label for e in self.entities}
        ends = {e.end_char: True for e in self.entities}
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
    doc = NER_Document(text)
    processed_text = doc.get_entities_with_markup()
    return render_template('result.html',processed_text=processed_text)

def sanity_check():
    text = "1/2 large sweet red onion, thinly sliced"
    doc = NER_Document(text)
    processed_text = doc.get_entities_with_markup()
    print(processed_text)

if __name__ == '__main__':
    app.run(debug=True)

# "1/2 cup fresh bean sprouts, or to taste (optional)"
# "4 eggs, whites and yolks separated"
