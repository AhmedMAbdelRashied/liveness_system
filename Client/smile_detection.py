import cv2
import numpy as np
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

smile_cc = cv2.CascadeClassifier(os.path.join(package_directory, 'haarcascade_smile.xml'))
def detect(roi):
    """takes gray roi of a face and returns whether a smile is detected or not

    Args:
        roi (grayscale image): the region of interest to check for smile

    Returns:
        str: 'Smile' or 'Neutral'
    """
    # X = cv2.resize(roi,(48,48)).reshape(1,48,48,1)
    # scores = model.predict(X)
    # return emts[np.argmax(scores)]
    bboxes = smile_cc.detectMultiScale(roi, 2, 34,minSize=(50,20))
    bboxes = None if type(bboxes) == tuple else bboxes[0]
    if bboxes is not None:
        bboxes[2:] += bboxes[:2]
    return bboxes
    

if __name__ == '__main__':
    pass
    # from cv2 import VideoCapture
    # import video_face_utils as futils 
    # import time
    # import matplotlib.pyplot as plt

    # modelFile = "E:\\Project\\res10_300x300_ssd_iter_140000.caffemodel"
    # configFile = "E:\\Project\\deploy.prototxt.txt"
    # net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    # cap = cv2.VideoCapture()
    # cap.open(0, cv2.CAP_DSHOW)
    # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f'{w}x{h}')
    # ret = True
    # prev = []
    # fc = 0
    # t2 = 0
    # face_queue = []
    
    # while(ret):
    #     ret, frame = cap.read()
    #     t1 = time.time()
    #     prev, face = futils.detect_face(frame, net, 0.7, prev, dims = (200, 200),flag=0,bbox_exp = 0.0, fill = 'constant')
    #     if not (face is None):
    #         bb = np.array(prev,'int') - 500
    #         roi = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #         roi = cv2.equalizeHist(roi)
    #         cv2.rectangle(frame, (bb[0], bb[1])
    #                         , (bb[2], bb[3]),
    #                         (255,0,0), 2)
    #         sbb = detect(roi)
    #         if(sbb is not None):
    #             cv2.rectangle(roi, (sbb[0], sbb[1])
    #                             , (sbb[2], sbb[3]),
    #                             (255,0,0), 2)
    #         t2 = (time.time() - t1) * 0.1 + 0.9 * t2
    #         cv2.putText(frame, f'{str(round(1/t2))} FPS',(0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2,cv2.LINE_AA)
    #         cv2.imshow('Cam',frame)
    #         cv2.imshow('roi',cv2.resize(roi,(480,480)))

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # IMG = cv2.imread('test_images/smile2.png',cv2.IMREAD_GRAYSCALE)
    # # import time
    # # t = time.time()
    # # detect(IMG)
    # # t = time.time() - t
    # # print(f'{round(1/(t+1e-6),1)} FPS :: FT {round(t*1000)}ms')