from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, render_template
from ner import SpacyDocument

app = Flask(__name__)

app.config['SECRET_KEY'] = 'fc3bb2a43ff1103895a4ee315ee27740'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_users.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Entity(db.Model):
    name = db.Column(db.String(50), unique=True, primary_key=True)
    frequency = db.Column(db.Integer, default=0)


def create_all():
    with app.app_context():
        db.create_all()


create_all()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        doc = SpacyDocument(text)
        entities_markup = doc.get_entities_with_markup()
        entities = doc.get_entities()
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



@app.route('/entities', methods=['GET', 'POST'])
def entities():
    entities = Entity.query.all()
    return render_template('entities.html', entities=entities)


if __name__ == '__main__':
    app.run(debug=True)
