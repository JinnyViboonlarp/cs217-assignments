import streamlit as st
import pandas as pd
import io
import spacy

nlp = spacy.load("en_core_web_sm")

default_text = (
        "When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

load_css("main.css")

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

def spacyprocessing(text):
    doc = SpacyDocument(text)
    st.markdown(str(doc.get_entities_with_markup()), unsafe_allow_html=True)
    
    def load_dataframe(entities):
        return pd.DataFrame(
            {
                "named entity": [entity[3] for entity in entities],
                "type": [entity[2] for entity in entities],
                "start index": [entity[0] for entity in entities],
                "end index": [entity[1] for entity in entities]
            }
        )

    df = load_dataframe(doc.get_entities())
    df

if __name__ == '__main__':
 
    text = st.text_input("Enter text for spaCy NER:")
    if(st.button('Use default text')):
        #st.success(default_text)
        spacyprocessing(default_text)
     
    if(st.button('Submit')):
        spacyprocessing(text)
