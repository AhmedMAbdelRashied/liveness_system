import time
import cv2
import numpy as np
import video_face_utils as futils
import socketio
import base64
import smile_detection
import side_profile
from scipy import spatial
from scipy.stats import entropy
import os
import threading

package_directory = os.path.dirname(os.path.abspath(__file__))

prompts = ['left',
           'right', 
           'smile']

modelFile = os.path.join(package_directory, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(package_directory, "deploy.prototxt.txt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
sio = socketio.Client()
ADDRESS = 'http://localhost:5000'
checking = False

@sio.on('face_found')
def success(data):
    global checking
    with threading.Lock():
        print(f"{data['name']} Logged in Database")
        checking = False
        sio.disconnect()


@sio.on('face_notfound')
def failure(data):
    global checking
    with threading.Lock():
        print(f"Face unrecognized")
        checking = False
        sio.disconnect()



@sio.on('liveness_score')
def liveness_retrieved(data):
    print(f"Liveness Scores:\n{data}")

def exp_crop(factor,img_bbox,mode=0):
    w = round((img_bbox[2] - img_bbox[0])*factor)
    h = round((img_bbox[3] - img_bbox[1])*factor)
    img_bbox[0] -= w
    img_bbox[1] -= h
    img_bbox[2] += w
    img_bbox[3] += h
    if mode == 0:
        img_bbox = np.vstack((img_bbox,np.zeros(4))).max(axis = 0)
    return img_bbox.astype('int')
    
    

def create_streamer(sio: socketio.Client, ADDRESS: str) -> None:
    global checking
    try:
        sio.connect(ADDRESS)
    except:
        print("Server is Offline")
        return
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'{w}x{h}')
    ret = True
    prev = []
    p_facehist = None
    p_idx = 0
    prev_p = ''
    t2 = 0
    cd = 15
    valid = False
    p_shuffle = np.random.choice(3,3,False)
    bypass = False
    checking = True
    while(checking):
        t1 = time.time()
        ret, frame = cap.read()
        frame = frame.copy()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if(bypass):
            
            cv2.putText(frame, f'Return to neutral position',(0,h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2,cv2.LINE_AA)
            cv2.imshow('Webcam Feed', frame)
            _, face = futils.detect_face(frame, net, 0.5, [], dims = (227, 227),flag=0,bbox_exp = 0.0, fill = 'constant',aspect_ratio_threshold=(0.4,1))
            if face is None:
                sio.emit('clear_queue')
                bypass = False
                p_idx = 0
                p_shuffle = np.random.choice(3,3,False)
            else:
                KLD, p_facehist = futils.get_KLD(face, p_facehist)
                if KLD > 0.3:
                    p_facehist = None
                    bypass = False
                    p_idx = 0
                    p_shuffle = np.random.choice(3,3,False)
                    continue
                if cd > 0:
                    cd-=1
                    continue
                frame_b64 = base64.b64encode(face)
                try:
                    sio.emit('check_liveness',{'frame': frame_b64})
                except:
                    checking = False
            continue
        
        prev, face = futils.detect_face(frame, net, 0.5, prev, dims = (227, 227),flag=0,bbox_exp = 0.0, fill = 'constant', aspect_ratio_threshold=(0.4,1))
        
        if not (face is None):
            
            
            KLD, p_facehist = futils.get_KLD(face, p_facehist)
            if KLD > 0.3:
                p_facehist = None
                valid = False
                continue

                
            if(not valid):
                p_idx = 0
                p_shuffle = np.random.choice(3,3,False)
                valid = True
            prompt = prompts[p_shuffle[p_idx]]
            cv2.putText(frame, f'Prompt: {prompt}',(0,h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),2,cv2.LINE_AA)

            bb = exp_crop(.5,np.array(prev,'int'),0)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            _,orientation = side_profile.face_orientation(
                cv2.resize(gray[bb[1]:bb[3],
                                bb[0]:bb[2]],
                           (100,100)))

            sbb = smile_detection.detect(
                cv2.resize(gray[prev[1]:prev[3],
                                prev[0]:prev[2]],
                           (100,100)))
            cv2.putText(frame, f'Orientation: {orientation[0]}',(0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2,cv2.LINE_AA)
            sc = (0,0,255)
            smiled = sbb is not None
            if smiled:
                sbb[0] = sbb[0] + prev[0]
                sbb[1] = sbb[1] + prev[1]
                sbb[2] = sbb[2] + prev[0]
                sbb[3] = sbb[3] + prev[1]
                sc = (0,255,0)
            cv2.putText(frame, f'Smiling!',(0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc,2,cv2.LINE_AA)

            cv2.rectangle(frame, (prev[0], prev[1])
                          , (prev[2], prev[3]),
                          (255,0,0), 2)
            if (prompt == 'smile' and smiled) or (prompt == orientation[0]):
                prev_p = prompt
                p_idx += 1
                if p_idx >= len(prompts):
                    bypass = True
                    cooldown = 12
            elif (prev_p == 'smile' and smiled) or (prev_p == orientation[0]) or orientation[0] == '':
                pass
            else:
                p_idx = 0
                p_shuffle = np.random.choice(3,3,False)
            # print(f'{bb}')
        else:
            valid = False
        t2 = (time.time() - t1) * 0.1 + 0.9 * t2
        cv2.putText(frame, f'{str(round(1/t2))} FPS',(0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255),2,cv2.LINE_AA)
        cv2.imshow('Webcam Feed', frame)
        
        
    cap.release()
    cv2.destroyAllWindows()
    
    
    
    


if __name__ == '__main__':
    while(True):
        choice = input('start? (y/n)')
        if choice.lower() == 'y':
            create_streamer(sio, ADDRESS)
        else: 
            break