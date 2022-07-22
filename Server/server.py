from flask import Flask, request, Response ,render_template,send_file,send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
from db import db_init, db
from models import *
import cv2
import numpy as np
import base64
from flask import Flask
from flask.globals import request
from flask_socketio import SocketIO
import base64
import numpy as np
import cv2
import os
import facenet_pretrained as facenet
import re
import liveness_model as liveness
#from tensorflow import keras 
package_directory = os.path.dirname(os.path.abspath(__file__))


from collections import defaultdict
q_dict = defaultdict(list)
r_dict = defaultdict(lambda: False)
day_q = []
app = Flask(__name__)

UPLOAD_FOLDER = 'images/'
app.config['SECRET_KEY'] = 'secret!'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app, async_handlers=True)
socketio.init_app(app, cors_allowed_origins="*")

facenet_model = facenet.get_model()
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    
    file = request.files['pic']
    if not file:
        return 'No pic uploaded!', 400
    name = request.form.get('name').replace(' ','_')
    path = os.path.join(app.config['UPLOAD_FOLDER'], f'{name}_{file.filename}')
    file.save(os.path.join(package_directory,path))
    face_enc = facenet.img_to_encoding(facenet_model, os.path.join(package_directory,path))
    if face_enc is None:
        return 'No face found!', 400
    np.save(os.path.splitext(path)[0]+'.npy', face_enc)
    person=Person(name=name,path=path)
    db.session.add(person)
    db.session.commit()

    return 'Img Uploaded!', 200

@app.route('/persons')
def get_img():
  x= Person.query.all()
  for person in x :
    print(person.name)
    print(person.path)
  return render_template('persons.html',data=x)

@app.route('/timelogs')
def get_timelogs():
  x= Day.query.all()
  for log in reversed(x):
    print(log.person_id)
    print(log.day)
  return render_template('times.html',data=x)

@app.route('/<int:id>')
def get_person_img(id):
    x= Person.query.filter_by(id=id).first()
    print(x.path)
    if not x:
        return 'Img Not Found!', 404
    return send_file(os.path.join(package_directory,x.path))

@app.route('/image/<string:path>')
def image(path):
  return send_file(os.path.join(package_directory, path))

@socketio.on('connect')
def test_connect():
  print(f'{request.sid} connected')

@socketio.on('clear_queue')
def clear_sid():
  sid = request.sid
  if sid in q_dict.keys():
    q_dict.pop(sid)
    r_dict.pop(sid)
  
@socketio.on('disconnect')
def disconnect():
  sid = request.sid
  print(f"{sid} disconnected")
  if sid in q_dict.keys():
    q_dict.pop(sid)
    r_dict.pop(sid)

@socketio.on('check_liveness')
def frame_handler(data):
  sid = request.sid
  frame_bytes = data['frame']
  buff = base64.decodebytes(frame_bytes)
  frame = np.frombuffer(buff, dtype=np.uint8).reshape(227,227,3)
  real_confidence = predict(frame, sid)
  if real_confidence > 0.7:
    r_dict[sid] = True
    frame = facenet.prepface(frame)
    match = get_id(frame)
    
    if match is None:
      socketio.emit('face_notfound', room=sid)
    else:
      if sid not in day_q:
        attendance = Day(person_id = match[0])
        db.session.add(attendance)
        db.session.commit()
        day_q.append(sid)
      socketio.emit('face_found', {'name':match[1]}, room=sid)

def get_id(frame):
  if frame is None:
    return None
  enc = facenet_model.predict(frame).ravel()
  x= Person.query.all()
  dists = []
  names = []
  ids = []
  for person in x:
    ids.append(person.id)
    names.append(person.name)
    enc_path = os.path.splitext(os.path.join(package_directory, person.path))[0] + '.npy'
    dist = facenet.findCosineDistance(enc, np.load(enc_path))
    dists.append(dist)
  min_idx = np.argmin(dists)
  print(dists)
  if dists[min_idx] < 0.07:
    return ids[min_idx], names[min_idx]
  return None
    

def predict(X, sid):
  sc_d  = liveness.get_scores(X)
  socketio.emit('liveness_score', str(sc_d), room=sid)
  print(sc_d)
  return sc_d
  
  
    
    

if __name__ == '__main__':
  socketio.run(app,debug=True)