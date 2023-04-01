import io
import spacy
import pickle
import pandas as pd
import os

from flask import Flask, jsonify, request, render_template

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

description = "This is a spacy NER service wrapped in Flask server. The service can \
perform tokenization, lemmatization, and NER (Named Entity Recognition) on the input text"

# This is the parts of the html page that shows the entity list
openpart_table_html = "<p>All entities seen so far:</p>\n"
endpart_table_html = """\n<p></p><form method = "POST">
  <input type="submit" name="btn" value="Redo NER">
</form><p></p><form method = "POST">
  <input type="submit" name="btn" value="Reset database">
</form>"""

# Loading database
database_path = "database.pickle"
if(os.path.exists(database_path)):
    with open(database_path, 'rb') as handle:
        entdict = pickle.load(handle)
else:
    entdict = {}

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

    # Update the dict storing entities with the new recognized ones
    def get_entities_dict(self, entdict):
        entities = []
        for e in self.doc.ents:
            et = e.text
            entdict[et] = (entdict.get(et, 0) + 1)
        return entdict

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

def create_html_from_database(entdict):
    # getting the dict of entities, create html page
    df = pd.DataFrame.from_dict(entdict, orient='index', columns=['instances'])
    html = df.to_html()
    text_file = open("templates/table.html", "w")
    text_file.write(openpart_table_html)
    text_file.write(html)
    text_file.write(endpart_table_html)
    text_file.close()
    return

# Note the methods argument, the default is a list with just GET, if you do not
# add POST then the resource will not accept POST requests.
@app.route('/')
def form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def form_post():
    global entdict
    if request.form['btn']=='Submit':
        
        text = str(request.form['inputtext'])
        doc = SpacyDocument(text)
        processed_text = doc.get_entities_with_markup()
        # Update the database and save it
        entdict = doc.get_entities_dict(entdict)
        with open(database_path, 'wb') as handle:
            pickle.dump(entdict, handle)
        return render_template('result.html',processed_text=processed_text)
    
    elif request.form['btn']=='Show list of entities so far (NOT including the unsubmitted text)' \
         or request.form['btn']=='Show list of entities so far':

        # Load database and write it into html file to be shown
        create_html_from_database(entdict)
        return render_template('table.html')
    
    elif request.form['btn']=='Redo NER':
        return render_template('form.html')

    elif request.form['btn']=='Reset database':
        entdict = {}
        with open(database_path, 'wb') as handle:
            pickle.dump(entdict, handle)
        create_html_from_database(entdict)
        return render_template('table.html')

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)
