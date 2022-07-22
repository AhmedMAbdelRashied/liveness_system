import cv2
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
package_directory = os.path.dirname(os.path.abspath(__file__))

side = cv2.CascadeClassifier(os.path.join(package_directory, 'haarcascade_profileface.xml'))

def get_areas(boxes):
    """
    Calculate Area(s) of one or more bounding boxes
    Args:
        boxes : an iterable of boxes where each bos is x0,y0,x1,y1

    Returns:
        list: areas of each box
    """
    areas = []
    for box in boxes:
        x0,y0,x1,y1 = box
        area = (y1-y0)*(x1-x0)
        areas.append(area)
    return areas

def detect(img, cascade):
    rects,_,confidence = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=7, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
    if len(rects) == 0:
        return (),()
    rects[:,2:] += rects[:,:2]
    return rects,confidence


def convert_rightbox(x_max,box_right):
    """_summary_

    Args:
        img (_type_): _description_
        box_right (_type_): _description_

    Returns:
        _type_: _description_
    """
    res = np.array([])
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max-box_[2]
        box[2] = x_max-box_[0]
        if res.size == 0:
            res = np.expand_dims(box,axis=0)
        else:
            res = np.vstack((res,box))
    return res



def face_orientation(gray):
    # left_face
    box_left, w_left = detect(gray,side)
    if len(box_left)==0:
        box_left = []
        name_left = []
    else:
        name_left = len(box_left)*["left"]
    # right_face
    gray_flipped = cv2.flip(gray, 1)
    box_right, w_right = detect(gray_flipped,side)
    if len(box_right)==0:
        box_right = []
        name_right = []
    else:
        width = gray.shape[1]
        box_right = convert_rightbox(width, box_right)
        name_right = len(box_right)*["right"]

    boxes = list(box_left)+list(box_right)
    names = list(name_left)+list(name_right)
    if len(boxes)==0:
        return boxes, ['']
    else:
        index = np.argmax(get_areas(boxes))
        boxes = [boxes[index].tolist()]
        names = [names[index]]
    return boxes, names
import time
if __name__ == '__main__':
    img = cv2.imread('test_images\smile2.png',cv2.IMREAD_GRAYSCALE)
    t = time.time()
    print(face_orientation(img))
    t = time.time() - t
    print(f'{round(1/t,1)} FPS :: FT {round(t*1000)}ms')
    