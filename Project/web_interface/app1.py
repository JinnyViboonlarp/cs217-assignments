import io

from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, render_template
from ner import SpacyDocument

app = Flask(__name__)

app.config['SECRET_KEY'] = 'fc3bb2a43ff1103895a4ee315ee27740'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_users.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

recent_info = None # storing info of the most recently processed text (input_text and entities)


class Entity(db.Model):
    name = db.Column(db.String(50), unique=True, primary_key=True)
    frequency = db.Column(db.Integer, default=0)


def create_all():
    with app.app_context():
        db.create_all()


create_all()

def create_markup(ner_info):
    (text, entities) = ner_info
    starts = {e[0]: e[2] for e in entities}
    #starts = {e.start_char: e.label for e in entities}
    ends = {e[1]: True for e in entities}
    #ends = {e.end_char: True for e in entities}
    buffer = io.StringIO()
    for p, char in enumerate(text):
        if p in ends:
            buffer.write('</entity>')
        if p in starts:
            buffer.write('<entity class="%s">' % starts[p])
        buffer.write(char)
    markup = buffer.getvalue()
    return '<markup>%s</markup>' % markup
    

@app.route('/', methods=['GET', 'POST'])
def index():
    global recent_info
    if request.method == 'POST':
        text = request.form['text']
        doc = SpacyDocument(text)
        entities_markup = doc.get_entities_with_markup()
        entities = doc.get_entities()
        # entities, for example, looks like this: [(0, 4, 'PERSON', 'Anna'), (43, 55, 'ORG', 'Moody Street')]
        recent_info = (text, entities)
        for entity in entities:
            entity_name = entity[3]
            db_entity = Entity.query.filter_by(name=entity_name).first()
            if db_entity:
                db_entity.frequency += 1
            else:
                db_entity = Entity(name=entity_name, frequency=1)
                db.session.add(db_entity)
            db.session.commit()
        return render_template('result.html', entities_markup=entities_markup, text=text, entities=entities)
    else:
        return render_template('form.html')
    

@app.route('/update', methods=['GET', 'POST'])
def update():
    global recent_info
    (text, entities) = recent_info
    updated_entities = []
    for entity in entities:
        (start, end, old_label, name) = entity
        new_label = request.form[name]
        updated_entities.append((start, end, new_label, name))
    recent_info = (text, updated_entities)
    entities_markup = create_markup(recent_info)
    return render_template('result.html', entities_markup=entities_markup, text=text, entities=updated_entities)

@app.route('/entities', methods=['GET', 'POST'])
def entities():
    entities = Entity.query.all()
    return render_template('entities.html', entities=entities)


if __name__ == '__main__':
    app.run(debug=True)
