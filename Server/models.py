from db import db
from datetime import datetime
import os



class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    path= db.Column(db.String(128),nullable=False)
    
class Day(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'),nullable=False)
    day=db.Column(db.DateTime, default=datetime.utcnow)