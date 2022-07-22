# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-07-02T09:04:13.730727Z","iopub.execute_input":"2022-07-02T09:04:13.731497Z","iopub.status.idle":"2022-07-02T09:04:13.773387Z","shell.execute_reply.started":"2022-07-02T09:04:13.731444Z","shell.execute_reply":"2022-07-02T09:04:13.772045Z"}}
import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.stats import entropy

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

def get_KLD(curr, p_facehist):
    gray_face = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
    if p_facehist is None:
        p_facehist = cv2.calcHist([gray_face],[0],None,[256],[0,256])+0.0001

    c_facehist = cv2.calcHist([gray_face],[0],None,[256],[0,256])+0.0001
    KLD = entropy(p_facehist, c_facehist)[0]
    return KLD, c_facehist

def read_vid(path, isColor = True):
    """
    read video from disk 
    """
    cap = cv2.VideoCapture(path)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.zeros((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, temp = cap.read()
        # import pdb; pdb.set_trace()
        if ret:
            buf[fc] = temp
        fc += 1
    cap.release()
    if not isColor:
        buf = buf[:,:,:,0]
    return buf

def write_vid(path, vid, isColor = True):
    """
    write to disk
    """
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (vid.shape[2], vid.shape[1]), isColor)
    [out.write(f) for f in vid]
    out.release()


def detect_face(image, net, min_confidence, prev=[], dims = (224, 224),flag=0,bbox_exp = 0.1, fill = 'reflect', aspect_ratio_threshold = None):
    """
    USING: OPENCV DNN SSD FACE MODEL
    get one(highest score) face from an image only if confidence > min confidence score
    also increases the bounding box area to get more back ground context
    the final face is cropped and resized to dims
    (224, 224) by default
    flag: 0 for cropping
          1 for drawing 
    dims are ignored if flag is set to 1
    bbox_exp: expand bounding box by a factor in all directions
    bounding box is calculated using moving average to reduct fluctuations
    aspect_ratio_threshold: a Tuple is expected (min, max) and face is rejected if aspect ratio isnot witihn the range set by the tuple 
    """
    image_width = image.shape[1]
    image_height = image.shape[0]
    preprocessed_image = cv2.dnn.blobFromImage(image, size=(300, 300))

    net.setInput(preprocessed_image)

    results = net.forward()    
    # print(results.shape)
    output_image = None
    coords = []
    for face in results[0][0]:
        # import pdb; pdb.set_trace()
        face_found = False
        # Retrieve the face detection confidence score.
        face_confidence = face[2]
        is_face = face[1]
        # Retrieve the bounding box of the face.
        bbox = face[3:]
        area = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))*image_height*image_width
        
        # Check if the face detection confidence score is greater than the thresold.
        if (bbox > 1).any() or (bbox<0).any():
            continue
        if (face_confidence > min_confidence) and (area > 100*100) and (is_face > 0):
#             import pdb; pdb.set_trace()
            if aspect_ratio_threshold is not None:
                AR = (bbox[2]-bbox[0])/(bbox[3]-bbox[1])
                if not (aspect_ratio_threshold[0] < AR < aspect_ratio_threshold[1]):
                    continue
            pad = 500
            w = (bbox[2]- bbox[0]) * image_width
            h = (bbox[3]- bbox[1]) * image_height
            # Retrieve the bounding box coordinates of the face and scale them according to the original size of the image.
            x1 = int((bbox[0] * image_width)  - (w*bbox_exp)) + pad
            y1 = int((bbox[1] * image_height) - (h*bbox_exp)) + pad
            x2 = int((bbox[2] * image_width)  + (w*bbox_exp)) + pad
            y2 = int((bbox[3] * image_height) + (h*bbox_exp)) + pad
            
            if len(prev) > 0:
                prev += 500
                alpha = 0.2
                x1= int(round(prev[0]*(1-alpha) + x1*alpha))
                y1= int(round(prev[1]*(1-alpha) + y1*alpha))
                x2= int(round(prev[2]*(1-alpha) + x2*alpha))
                y2= int(round(prev[3]*(1-alpha) + y2*alpha))
            # size_list.append((x1,y1,x2,y2))
            if flag:
                output_image = image.copy()
                cv2.rectangle(output_image, (x1-pad, y1-pad), (x2-pad, y2-pad), (255,0,0), 2)
            else:
                output_image = cv2.resize(np.pad(image,((pad,pad), (pad,pad), (0,0)), mode=fill)[y1:y2,x1:x2], dims)
            face_found = True
            coords = np.array([x1,y1,x2,y2]) - 500
            break
    # if not face_found:
    #     # import pdb; pdb.set_trace()
    #     bbox = results[0][0][0][3:]
    #     area = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))*image_height*image_width
    #     # print(f'no face found {area}<{32*32}, \n{results[0][0][:2]}');
    # # return output_image 
    # # print(face_confidence)
    return coords, output_image#, face_found

def detect_vid_wrapper(video, net, min_confidence, dims = (224, 224), flag=0, bbox_exp = 0.1, fill = 'reflect'):
    """
    wrapper to iterate over the video and detect the faces frame by frame
    flag 0 crop
    flag 1 draw
    """
    # op_vid = np.zeros((video.shape[0],64,64,3),dtype='uint8')
    frames_list = []
    success = False
    prev = []
    for i in range(video.shape[0]):
        prev, frame = detect_face(video[i], net, min_confidence, prev, dims, flag, bbox_exp, fill)
        if frame is not None:
            frames_list.append(frame)
    op_vid = np.asarray(frames_list, dtype='uint8')
    return op_vid

def vid_yuv_to_bgr(video):
    """
    transform video from YUV to BGR(Opencv format) to be used in further processing
    """
    result_video = np.zeros(shape = video.shape)
    for i in range(video.shape[0]):
        result_video[i] = cv2.cvtColor(video[i], cv2.COLOR_YUV2BGR)
    return result_video.astype('uint8')

def vid_bgr_to_yuv(video):
    """
    transform video from BGR(Opencv format) to YUV to be used in eqHist
    """
    result_video = np.zeros(shape = video.shape)
    for i in range(video.shape[0]):
        result_video[i] = cv2.cvtColor(video[i], cv2.COLOR_BGR2YUV)
    return result_video.astype('uint8')

def eqHist(video,):
    """
    equalize histogram of the lumina channel Y of the video 
    the equalization is done frame by frame
    """
#     shape = video.shape
#     result = vid_bgr_to_yuv(video)
#     result[:,:,:,0] = cv2.equalizeHist(result[:,:,:,0].reshape(-1,1)).reshape(result[:,:,:,0].shape)
    frame_count = video.shape[0] 
    for i in range(frame_count):
        cv2.cvtColor(video[i], cv2.COLOR_BGR2YUV, video[i])
        channels = cv2.split(video[i])
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, video[i])
        cv2.cvtColor(video[i], cv2.COLOR_YUV2BGR, video[i])
    return video
#     return vid_yuv_to_bgr(result.astype('uint8'))

def bp_eqrple_filt(fs = 25.0, band = [0.7, 2.6], trans_width = 0.1, numtaps = 800):
    """
    function to create an approximation of equirpple filter
    """
    edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
    taps = signal.remez(numtaps, edges, [0, 1, 0], fs=fs)
    return taps

def apply_filter(X,taps,):
    """
    function to apply filter to video by fast fourier transfort convolution
    """
    Eq_filter = np.expand_dims(taps,axis=(1,2,3))
    f_X = signal.fftconvolve(X, Eq_filter,axes = 0, mode='same')
    return f_X

def ST_Mapping(X, fs):
    """
    Get the energy mapping of the filtered video
    the final output is a 2D Map
    """
    freqs, ESD = signal.welch(X, fs=fs, axis = 0, scaling='spectrum')

    ESD = ESD.sum(axis=0)
    e_map = np.sqrt(np.square(ESD).sum(axis = -1))
    return e_map

def freq_analysis_wrapper(X, taps, fs=25):
    """
    wrapper function to apply both frequency analysis functions to the video and return the map
    """
    F = apply_filter(X,taps)
    return ST_Mapping(F,fs)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-07-02T09:02:59.486190Z","iopub.execute_input":"2022-07-02T09:02:59.486919Z","iopub.status.idle":"2022-07-02T09:03:26.563324Z","shell.execute_reply.started":"2022-07-02T09:02:59.486876Z","shell.execute_reply":"2022-07-02T09:03:26.561925Z"}}
# if __name__ == '__main__':
#     !pip install gdown -q
#     !gdown 1H9sg0daYdvMB7_F1o7g70fbla1PmVVq4
#     !gdown 1dEOIB10Q9UgOdzbb_ed_ad7egmNiURi0
#     !gdown 124OvotW6PsuJIvkAiZwGW5WKct7egyjn
#     modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
#     configFile = "deploy.prototxt.txt"
#     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-07-02T09:04:16.772121Z","iopub.execute_input":"2022-07-02T09:04:16.772502Z","iopub.status.idle":"2022-07-02T09:04:31.680745Z","shell.execute_reply.started":"2022-07-02T09:04:16.772469Z","shell.execute_reply":"2022-07-02T09:04:31.679600Z"}}
# if __name__ == '__main__':
#     taps = bp_eqrple_filt(fs=28)
#     X = read_vid('VID_20220702_070138.mp4')
#     X = detect_vid_wrapper(X, net, 0.5, dims=(128, 128))
#     video = eqHist(X)
# #     import pdb; pdb.set_trace()
#     Map = freq_analysis_wrapper(video, taps)

#     import matplotlib.pyplot as plt

#     plt.imshow(Map)

#     Map.max()

#     Eq_filter = np.expand_dims(taps,axis=(1,2,3))
#     f_X = signal.fftconvolve(X, Eq_filter,axes = 0, mode='same')
#     plt.plot(f_X[:,15,15,0])

#     sos = signal.butter(50, [0.5,2.6], 'bandpass', analog=False, output='sos', fs =30)
#     f_X2 = signal.sosfilt(sos,X,axis = 0)
#     plt.plot(f_X2[:,15,15,0])

#     FFT = fft(X,axis = 0)
#     plt.plot(FFT[:,15,15,0])

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:35:38.221445Z","iopub.execute_input":"2022-07-01T06:35:38.221768Z","iopub.status.idle":"2022-07-01T06:35:38.730655Z","shell.execute_reply.started":"2022-07-01T06:35:38.221739Z","shell.execute_reply":"2022-07-01T06:35:38.729591Z"}}
#     FFT = fft(f_X,axis = 0)
#     plt.plot(FFT[:,15,15,0])

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:35:38.734513Z","iopub.execute_input":"2022-07-01T06:35:38.734869Z","iopub.status.idle":"2022-07-01T06:35:39.171186Z","shell.execute_reply.started":"2022-07-01T06:35:38.734836Z","shell.execute_reply":"2022-07-01T06:35:39.170244Z"}}
#     FFT = fft(f_X2,axis = 0)
#     plt.plot(FFT[:,15,15,0])

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:35:39.174670Z","iopub.execute_input":"2022-07-01T06:35:39.175025Z","iopub.status.idle":"2022-07-01T06:35:40.025529Z","shell.execute_reply.started":"2022-07-01T06:35:39.174995Z","shell.execute_reply":"2022-07-01T06:35:40.024225Z"}}
#     freqs, w = signal.welch(X,fs=28,nperseg=28,axis = 0,scaling='spectrum')
#     w = np.sqrt(np.square(w).sum(axis = -1))
#     plt.semilogy(freqs, w[:,15,15])
#     # plt.ylim([0.5e-3, 1])
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('ESD [V**2]')
#     plt.show()

# cv2.VideoCapture('VID_20220702_070138.mp4').get(cv2.CAP_PROP_FPS)

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:37:16.578429Z","iopub.execute_input":"2022-07-01T06:37:16.578836Z","iopub.status.idle":"2022-07-01T06:37:17.242474Z","shell.execute_reply.started":"2022-07-01T06:37:16.578802Z","shell.execute_reply":"2022-07-01T06:37:17.241372Z"}}
#     idx = np.argwhere((freqs >= 0.7)&(freqs <=2.6))
#     MAP1 = w[idx].sum(axis = 0).reshape(128,128)
#     plt.imshow(MAP1)

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:36:04.857817Z","iopub.execute_input":"2022-07-01T06:36:04.858227Z","iopub.status.idle":"2022-07-01T06:36:05.862863Z","shell.execute_reply.started":"2022-07-01T06:36:04.858192Z","shell.execute_reply":"2022-07-01T06:36:05.862038Z"}}
#     freqs, w = signal.welch(f_X,fs=28,nperseg=28,axis = 0,scaling='spectrum')
#     w = np.sqrt(np.square(w).sum(axis = -1))
#     plt.semilogy(freqs, w[:,15,15])
#     # plt.ylim([0.5e-3, 1])
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('ESD [V**2]')
#     plt.show()

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:42:08.450864Z","iopub.execute_input":"2022-07-01T06:42:08.451751Z","iopub.status.idle":"2022-07-01T06:42:08.627405Z","shell.execute_reply.started":"2022-07-01T06:42:08.451705Z","shell.execute_reply":"2022-07-01T06:42:08.626279Z"}}

# #     idx = np.argwhere((freqs >= 0.7)&(freqs <=2.6))
#     MAP3 = w.sum(axis = 0).reshape(128,128)
#     MAP3 = (MAP3 - MAP3.mean())/MAP3.std()
#     plt.imshow(MAP2)



#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:36:20.092789Z","iopub.execute_input":"2022-07-01T06:36:20.093190Z","iopub.status.idle":"2022-07-01T06:36:20.794155Z","shell.execute_reply.started":"2022-07-01T06:36:20.093156Z","shell.execute_reply":"2022-07-01T06:36:20.793391Z"}}
#     freqs, w = signal.welch(f_X2,fs=28,nperseg=28,axis = 0,scaling='spectrum')
#     w = np.sqrt(np.square(w).sum(axis = -1))
#     plt.semilogy(freqs, w[:,15,15])
#     # plt.ylim([0.5e-3, 1])
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('ESD [V**2]')
#     plt.show()

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:36:21.209586Z","iopub.execute_input":"2022-07-01T06:36:21.210241Z","iopub.status.idle":"2022-07-01T06:36:21.421515Z","shell.execute_reply.started":"2022-07-01T06:36:21.210205Z","shell.execute_reply":"2022-07-01T06:36:21.420369Z"}}
#     idx = np.argwhere((freqs >= 0.7)&(freqs <=2.6))
#     MAP3 = w[idx].sum(axis = 0).reshape(128,128)
#     plt.imshow(MAP3)

#     # %% [code] {"execution":{"iopub.status.busy":"2022-07-01T06:36:54.355311Z","iopub.execute_input":"2022-07-01T06:36:54.355739Z","iopub.status.idle":"2022-07-01T06:36:54.565236Z","shell.execute_reply.started":"2022-07-01T06:36:54.355701Z","shell.execute_reply":"2022-07-01T06:36:54.564081Z"}}
#     plt.imshow(Map)

#     # %% [code]
#     Map

# %% [code] {"execution":{"iopub.status.busy":"2022-07-02T06:01:04.183988Z","iopub.execute_input":"2022-07-02T06:01:04.184714Z","iopub.status.idle":"2022-07-02T06:01:04.192610Z","shell.execute_reply.started":"2022-07-02T06:01:04.184665Z","shell.execute_reply":"2022-07-02T06:01:04.191401Z"},"jupyter":{"outputs_hidden":false}}
# plt.semilogy(f, ESD[:,15,15]);plt.xlabel('frequency [Hz]');plt.ylabel('ESD [V**2]');plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
