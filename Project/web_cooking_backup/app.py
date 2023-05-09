from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, render_template, url_for, redirect
from cooking_ner import NER_Document

app = Flask(__name__)

app.config['SECRET_KEY'] = 'fc3bb2a43ff1103895a4ee315ee27740'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_users.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

doc = None # storing info of the most recently processed text


class Entity(db.Model):
    name = db.Column(db.String(50), unique=True, primary_key=True)
    frequency = db.Column(db.Integer, default=0)


def create_all():
    with app.app_context():
        db.create_all()


create_all()

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def form_submit():
    return redirect(url_for('form'))

@app.route('/', methods=['POST'])
def form_post():
    global doc
    text = request.form.get('text','')
    doc = NER_Document(text)
    entities_markup = doc.get_entities_with_markup()
    # entities is a list of 'entity', and each entity has property: start_char, end_char, label, text
    for entity in doc.entities:
        db_entity = Entity.query.filter_by(name=entity.text).first()
        if db_entity:
            db_entity.frequency += 1
        else:
            db_entity = Entity(name=entity.text, frequency=1)
            db.session.add(db_entity)
        db.session.commit()
    return render_template('result.html', entities_markup=entities_markup, text=doc.text, entities=doc.entities)

@app.route('/update', methods=['GET', 'POST'])
def update():
    global doc
    for entity in doc.entities:
        entity.label = request.form[entity.text]
    entities_markup = doc.get_entities_with_markup()
    return render_template('result.html', entities_markup=entities_markup, text=doc.text, entities=doc.entities)

@app.route('/entities', methods=['GET', 'POST'])
def entities():
    entities = Entity.query.all()
    return render_template('entities.html', entities=entities)


if __name__ == '__main__':
    app.run(debug=True)

# "1/2 cup fresh bean sprouts, or to taste (optional)"
# "4 eggs, whites and yolks separated"
