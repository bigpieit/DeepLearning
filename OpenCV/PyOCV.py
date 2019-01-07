import cv2 as cv
import numpy as np
import sys, os
import dlib

def day1():
    imC = cv.imread("./data/lena.jpg", 1)
    print("imC row: ", imC.shape[0], " col : ", imC.shape[1], " channel : ", imC.shape[2])
    print("imC data type : ", imC.dtype, "imC type : ", type(imC))
    imG = cv.imread("./data/lena.jpg", 0)
    print("imG row: ", imG.shape[0], " col : ", imG.shape[1], " total shape size : ",
          len(imG.shape))
    print("imG data type : ", imG.dtype, "imG type : ", type(imG))


    # basic ops
    # Read, Write & Display
    image = cv.imread("./data/lena.png", 1)
    if image is None:
        print("Damn! Image not there")
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite("./result/imageGray.jpg", grayImage)
    #cv.namedWindow("image", cv.WINDOW_AUTOSIZE) # unnecessary
    cv.imshow("image", image)
    cv.waitKey(0)
    #cv.namedWindow("gray image", cv.WINDOW_NORMAL) # unnecessary
    cv.imshow("gray image",grayImage)
    cv.waitKey(0)

    # Resize & Crop
    source = cv.imread("./data/lena.png", 1)
    scaleX = 0.6; scaleY = 0.6
    scaleDown = cv.resize(source, None, fx=scaleX,   fy=scaleY,   interpolation=cv.INTER_LINEAR)
    scaleUp   = cv.resize(source, None, fx=scaleX*3, fy=scaleY*3, interpolation=cv.INTER_LINEAR)
    crop = source[50:150, 20:200]
    cv.imshow("Original", source)
    cv.imshow("Scaled Down", scaleDown)
    cv.imshow("Scaled Up", scaleUp)
    cv.imshow("Cropped Image", crop)
    cv.waitKey(0)

    # Rotate
    source = cv.imread("./data/lena.png", 1)
    dim = source.shape
    rotationAngle = -30
    scaleFactor = 1
    rotationMatrix10= cv.getRotationMatrix2D((dim[1]/2, dim[0]/2), rotationAngle, scaleFactor)
    print("rotation matrix10 : \n", rotationMatrix10)
    result = cv.warpAffine(source, rotationMatrix10, (dim[1], dim[0]))
    cv.imshow("Original", source)
    cv.imshow("Rotated Image", result)
    cv.waitKey(0)

    a = np.zeros((2, 3))
    print("a: \n",a)
    b = np.ones((2, 3))
    print("b: \n",b)
    c = np.eye(2)
    print("c: \n",c)

def day1_1():
    image = cv.imread("./data/lena.png")

    # line
    imageLine = image.copy()
    cv.line(imageLine, (322, 179), (400, 400), (0,255,0), thickness=2, lineType=cv.LINE_AA)
    cv.imshow("imageLine", imageLine)
    cv.imwrite("./result/imageLine.png", imageLine)
    cv.waitKey(0)

    # circle
    imageCircle = image.copy()
    cv.circle(imageCircle, (290,285), 100, (0,255,0), thickness=2, lineType=cv.LINE_AA)
    cv.imshow("imageCircle", imageCircle)
    cv.imwrite("./result/imageCircle.png", imageCircle)
    cv.waitKey(0)

    # Draw an ellipse
    # IMP Note: Ellipse Centers and Axis lengths must be integers
    imageEllipse = image.copy()
    cv.ellipse(imageEllipse, (360, 200), (100, 170),  45, 0, 180, (255, 0, 0), thickness=2,
               lineType=cv.LINE_AA)
    cv.ellipse(imageEllipse, (370, 190), (110, 160),  45, 0,  150, (0, 0, 0), thickness=2,
               lineType=cv.LINE_AA)
    cv.ellipse(imageEllipse, (360, 200), (100, 170), 135, 0, 360, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
    cv.imshow("ellipse", imageEllipse)
    cv.imwrite("imageEllipse.jpg", imageEllipse)
    cv.waitKey(0)

    # Draw a rectangle (thickness is a positive integer)
    imageRectangle = image.copy()
    cv.rectangle(imageRectangle, (208, 55), (450, 355), (0, 255, 0), thickness=2, lineType=cv.LINE_8)
    cv.imshow("rectangle", imageRectangle)
    cv.imwrite("imageRectangle.jpg", imageRectangle)

    # Put text into image
    imageText = image.copy()
    cv.putText(imageText, "Lena", (205, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("text", imageText)
    cv.imwrite("imageText.jpg", imageText)
    cv.waitKey(0)

# Lists to store the points
center=[]
circumference=[]
source_draw=None
def day1_drawCircle(action, x, y, flags, userdata):
    # Referencing global variables
    import math
    global center, circumference, source_draw
    # Action to be taken when left mouse button is pressed
    if action==cv.EVENT_LBUTTONDOWN:
        center=[(x,y)]
        # Mark the center
        cv.circle(source_draw, center[0], 1, (255,255,0), 2, cv.LINE_AA)
    # Action to be taken when left mouse button is released
    elif action==cv.EVENT_LBUTTONUP:
        circumference=[(x,y)]
        # Calculate radius of the circle
        radius = math.sqrt(math.pow(center[0][0]-circumference[0][0],2)
                    +math.pow(center[0][1]-circumference[0][1],2))
        # Draw the circle
        cv.circle(source_draw, center[0], int(radius), (0,255,0),2, cv.LINE_AA)
        cv.imshow("Window",source_draw)

def day1_2(strings):
    global source_draw
    if strings == "draw":
        source_draw = cv.imread("./data/lena.png", 1)
        # Make a dummy image, will be useful to clear the drawing
        dummy = source_draw.copy()
        cv.namedWindow("Window")
        # highgui function called when mouse events occur
        cv.setMouseCallback("Window", day1_drawCircle)
        k = 0
        # loop until escape character is pressed
        while k != 27:
            cv.imshow("Window", source_draw)
            cv.putText(source_draw, "Choose center, and drag, Press ESC to exit and c to clear",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            k = cv.waitKey(20) & 0xFF
            # Another way of cloning
            if k == 99:
                source_draw = dummy.copy()
        cv.destroyAllWindows()

def day3():
    # WarpAffine
    source = cv.imread("./data/lena.png",1)
    warpMat1 = np.float32([[ 1.2, 0.2, 2.0],
                         [-0.3, 1.3, 1.0]])
    warpMat2 = np.float32([[ 1.2, 0.3, 2.0],
                         [ 0.2, 1.3, 1.0]])
    result1 = cv.warpAffine(source, warpMat1,
                            (int(1.5*source.shape[0]), int(1.5*source.shape[1])),
                            None, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT_101)
    result2 = cv.warpAffine(source, warpMat2,
                            (int(1.5*source.shape[0]), int(1.5*source.shape[1])),
                            None, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT_101)
    cv.imshow("Source",  source)
    cv.imshow("Result1", result1)
    cv.imshow("Result2", result2)
    cv.waitKey(0)

def day3_getAffine():
    input  = np.float32([[50, 50], [100, 100], [200, 150]])
    output1 = np.float32([[72, 51], [142, 101], [272, 136]])
    output2 = np.float32([[77, 76], [152, 151], [287, 236]])
    warpMat1 = cv.getAffineTransform(input, output1)
    warpMat2 = cv.getAffineTransform(input, output2)
    print("Warp Matrix 1 : \n {} \n".format(warpMat1))
    print("Warp Matrix 2 : \n {} \n".format(warpMat2))

def day3_homography():
    dst = cv.imread("./data/book1.jpg", 1)
    ptr_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

    src = cv.imread("./data/book2.jpg", 1)
    ptr_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])

    h, status = cv.findHomography(ptr_src, ptr_dst)
    output = cv.warpPerspective(src, h, (dst.shape[1],  dst.shape[0]))
    cv.imshow("Source Image", src)
    cv.imshow("Dst Image", dst)
    cv.imshow("Warped Output Image", output)
    cv.waitKey(0)

def day4(name="sl"):
    faceCascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    faceNeighborsMax = 10
    neighborStep = 1


    src = cv.imread("./data/"+name+".jpg")
    if name == "sl":
        scaleX = 0.5
        scaleY = 0.5
        frame = cv.resize(src, None, fx=scaleX, fy=scaleY, interpolation=cv.INTER_LINEAR)
    else:
        frame = src.copy()


    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for neigh in range(1, faceNeighborsMax, neighborStep):
        faces = faceCascade.detectMultiScale(frameGray, scaleFactor=1.2, minNeighbors=neigh)
        frameClone = np.copy(frame)

        for (x,y,w,h) in faces:
            cv.rectangle(frameClone, (x,y), (x+w, y+h), (255, 0, 0), 2)

        cv.putText(img=frameClone, text="# Neighbors = {}".format(neigh), org=(10, 50), \
        fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=4)
        cv.imshow("Face Detection Demo", frameClone)
        if cv.waitKey(2000) & 0xFF == 27:
            cv.destroyAllWindows()
            sys.exit()

def day4_smile(name="sl"):
    faceCascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    smileCascade = cv.CascadeClassifier("models/haarcascade_smile.xml")
    smileNeighborsMax = 100
    neighborStep = 2

    src = cv.imread("./data/"+name+".jpg")
    if name == "sl":
        scaleX = 0.5
        scaleY = 0.5
        frame = cv.resize(src, None, fx=scaleX, fy=scaleY, interpolation=cv.INTER_LINEAR)
    else:
        frame = src.copy()

    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frameGray, scaleFactor=1.4, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        faceRoiGray = frameGray[y:y+h, x:x+w]

        for neigh in range(1, smileNeighborsMax, neighborStep):
            smile = smileCascade.detectMultiScale(faceRoiGray, scaleFactor=1.5, minNeighbors=neigh)
            frameClone = np.copy(frame)
            faceRoiClone = frameClone[y: y + h, x: x + w]
            for (xx, yy, ww, hh) in smile:
                cv.rectangle(faceRoiClone, (xx, yy), (xx+ww, yy+hh), (0, 255, 0), 2)

            cv.putText(img=frameClone, text="# Neighbors = {}".format(neigh), org=(10, 50),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=4)
            cv.imshow("Face and Smile Demo", frameClone)
            if cv.waitKey(500) & 0xFF == 27:
                cv.destroyAllWindows()
                sys.exit()

def day4_dnn_face(name="sl"):
    DNN = "TF"
    conf_threshold = 0.7

    if DNN == "CAFFE":
        modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "./models/deploy.prototxt"
        net = cv.dnn.readNetFromCaffe(caffeModel=modelFile, prototxt=configFile)
    else:
        modelFile = "./models/opencv_face_detector_uint8.pb"
        configFile = "./models/opencv_face_detector.pbtxt"
        net = cv.dnn.readNetFromTensorflow(model=modelFile, config=configFile)

    src = cv.imread("./data/"+name+".jpg", 1)
    if name == "sl":
        scaleX = 0.5
        scaleY = 0.5
        frame = cv.resize(src, None, fx=scaleX, fy=scaleY, interpolation=cv.INTER_LINEAR)
    else:
        frame = src.copy()

    height = frame.shape[0]
    width = frame.shape[1]
    meanB = int(np.mean(frame[:,:,0]))
    meanG = int(np.mean(frame[:,:,1]))
    meanR = int(np.mean(frame[:,:,2]))
    print("src BGR mean {} {} {}".format(meanB, meanG, meanR))

    # reducing output size missed lena 200->150 0.99->0.94
    # reducing output size 300->100 remove nuisance in HC
    # SL is always good
    blob = cv.dnn.blobFromImage(image=frame, scalefactor=1.0, size=(150, 150),
                                mean=[meanR, meanG, meanB], swapRB=False, crop=False)
    net.setInput(blob)
    detections =net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*width)
            y1 = int(detections[0,0,i,4]*height)
            x2 = int(detections[0,0,i,5]*width)
            y2 = int(detections[0,0,i,6]*height)
            cv.rectangle(img=frame, pt1=(x1,y1), pt2=(x2,y2), color=[0,255,0], thickness=int(
                height/150), lineType=cv.LINE_8)
            bboxes.append([confidence, x1,y1,x2,y2])
            print("NO.{} faces confidence: {}".format(i, confidence))

    cv.imshow("detected faces", frame)
    cv.waitKey()


SZ=20
CLASS_N=10
import itertools as it
def day5OR_HOGSVM():
    def split2d(img, cell_size, flatten=True):
        h, w = img.shape[:2]
        sx, sy = cell_size
        cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
        cells = np.array(cells)
        if flatten:
            cells = cells.reshape(-1, sy, sx)
        return cells

    def load_digits(fn):
        digits_img = cv.imread(fn, 0)
        digits = split2d(digits_img, (SZ, SZ))
        labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
        return digits, labels

    def deskew(img):
        # moments comes in handy while calculating useful information like centroid, area,
        # skewness of simple images with black backgrounds
        m = cv.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        # print("moments is :", m)
        skew = m["mu11"]/m["mu02"]
        M = np.float32([[1, skew, -0.5*skew*SZ], [0, 1, 0]])
        img = cv.warpAffine(img, M, dsize=(SZ, SZ), flags=cv.WARP_INVERSE_MAP |
                                                                cv.INTER_LINEAR)
        # cv.imshow("input", img)
        # cv.imshow("out", skewed_img)
        # cv.waitKey()
        return img

    def get_hog():
        winSize = (20, 20)#calculate one descriptor for the entire image.
        # Typically blockSize is set to 2 x cellSize tackle illumination but
        # not a problem here
        blockSize = (8, 8)
        blockStride = (4, 4)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True

        hog = cv.HOGDescriptor(winSize, blockSize,blockStride,cellSize,nbins, derivAperture,
                               winSigma,histogramNormType,L2HysThreshold, gammaCorrection,nlevels,
                               signedGradients)

        return hog

    def svmInit(C=12.5, gamma=0.50625):
        model = cv.ml.SVM_create()
        model.setGamma(gamma)
        model.setC(C)
        model.setKernel(cv.ml.SVM_RBF)
        model.setType(cv.ml.SVM_C_SVC)
        return model

    def svmTrain(model, samples, labels):
        model.train(samples, cv.ml.ROW_SAMPLE, labels)
        return model

    def svmEvaluate(model, digits, samples, labels):
        def grouper(n, iterable, fillvalue=None):
            '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
            args = [iter(iterable)] * n
            output = it.zip_longest(fillvalue=fillvalue, *args)
            return output

        def mosaic(w, imgs):
            '''Make a grid from images.

            w    -- number of grid columns
            imgs -- images (must have same size and format)
            '''
            imgs = iter(imgs)
            img0 = imgs.__next__()
            pad = np.zeros_like(img0)
            imgs = it.chain([img0], imgs)
            rows = grouper(w, imgs, pad)
            return np.vstack(map(np.hstack, rows))

        predictions = model.predict(samples)[1].ravel()
        accuracy = (labels == predictions).mean()
        print('Percentage Accuracy : %.4f %%'%(accuracy*100))

        confusion = np.zeros((10, 10), np.int32)
        for i, j in zip(labels, predictions):
            confusion[int(i), int(j)] += 1
        print("confusion matrix:\n", confusion)

        vis = []
        mispred = []
        for img, flag in zip(digits, predictions==labels):
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if not flag:
                img[..., :2] = 0
            vis.append(img)
        print("vis length : ", len(vis))
        print("pred   is : ", [int(dummy) for dummy in predictions[predictions!=labels]])
        print("labels is : ", [int(dummy) for dummy in labels[predictions!=labels]])
        return mosaic(20, vis)

    print("loading datat from digits.png ...")
    digits, label = load_digits("./data/digits.png")
    print("digits shape : ", digits.shape)

    print("shuffle data ...")
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, label = digits[shuffle], label[shuffle]

    print("deskew imaes ...")
    digits_deskewed = list(map(deskew, digits))

    print("get hog parameters ...")
    hog = get_hog()

    print("calculatiing hog descriptor for every image ...")
    hog_descriptors = []
    for idx, img in enumerate(digits_deskewed):
        if idx%500 == 0:
            print("{} / {} finished ...".format(idx, len(digits_deskewed)))
        hog_descriptors.append(hog.compute(img))

    print("b4 squeeze, total  length : ", len(hog_descriptors))
    print("b4 squeeze, id [0] length : ", len(hog_descriptors[0])) # 144 []s
    # print("b4 squeeze, id [0] : ", hog_descriptors[0])
    hog_descriptors = np.squeeze(hog_descriptors)
    print("af squeeze, length : ", len(hog_descriptors))
    print("af squeeze, id [0] length : ", len(hog_descriptors[0])) # [] of 144 elements
    # print("b4 squeeze, id [0] : ", hog_descriptors[0])


    print("spliting data 90% for traing 10% for testing ...")
    train_n = int(0.9 *len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(label, [train_n])

    print("training svm ...")
    model = svmInit()
    svmTrain(model, hog_descriptors_train, labels_train)

    print("evaluating model ...")
    vis = svmEvaluate(model, digits_test, hog_descriptors_test, labels_test)
    cv.imwrite("digits-classification.jpg",vis)
    cv.imshow("Vis", vis)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # skewed = deskew(digits[0,:,:])
    # scaleX = 10; scaleY = 10
    # scaleUp = cv.resize(skewed, None, fx=scaleX, fy=scaleY, interpolation=cv.INTER_LINEAR)
    # cv.imshow("first one",scaleUp)
    # cv.waitKey()

def day5_DL():
    weightFile ="models/tensorflow_inception_graph.pb"
    frame = cv.imread("data/panda.jpg")
    classFile = "models/imagenet_comp_graph_label_strings.txt"

    classes = None
    with open(classFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    inHeight = 224
    inWidth = 224
    swap_rgb = True
    mean = [117, 117, 117]
    scale = 1.0

    # Load Network
    net = cv.dnn.readNetFromTensorflow(weightFile)
    blob = cv.dnn.blobFromImage(frame, scalefactor=scale, size=(inWidth, inHeight), mean=mean,
                                swapRB=swap_rgb, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.flatten()
    classID = np.argmax(out)
    className = classes[classID]
    confidence = out[classID]
    print(classID)
    label = "Predicted Class = {}, Confidence = {:.3f}".format(className, confidence)
    print(label)
    cv.putText(frame, label, org=(10,30), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
               color=(0,0,255),thickness=2, lineType=cv.LINE_AA)
    cv.imshow("Classification Output", frame)
    cv.imwrite("day5_tfout.jpg", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

def day6_YOLO3(path, IsIMG=True):
    # Get the names of the output layers
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    # Initialize the parameters
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416  # Height of network's input image
    print("USAGE :")
    print('''
    For image : python object_detection_yolo.py --image=bird.jpg
    For Video : python object_detection_yolo.py --video=run.mp4
        ''')

    # Load names of classes
    classesFile = "./models/coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "./models/yolov3.cfg"
    modelWeights = "./models/yolov3.weights"

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Process inputs
    winName = 'Deep learning object detection in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    outputFile = "yolo_out_py.avi"
    if (IsIMG):
        # Open the image file
        if not os.path.isfile(path):
            print("Input image file ", path, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(path)
        outputFile = path[:-4]+'_yolo_out_py.jpg'
    else:
        # Open the video file
        if not os.path.isfile(path):
            print("Input video file ", path, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(path)
        outputFile = path[:-4]+'_yolo_out_py.avi'
    # else:
    #     # Webcam input
    #     cap = cv.VideoCapture(0)

    if (not IsIMG):
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (IsIMG):
            cv.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv.imshow(winName, frame)

def day7_FaceLandmark():
    def writeLandmarksToFile(landmarks, landmarksFileName):
        with open(landmarksFileName, 'w') as f:
            for p in landmarks.parts():
                f.write("%s %s\n" % (int(p.x), int(p.y)))

        f.close()
    # Draw the Landmarks on top of the face.
    def drawLandmarks(im, landmarks):
        for i, part in enumerate(landmarks.parts()):
            px = int(part.x)
            py = int(part.y)
            cv.circle(im, (px, py), 1, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
            cv.putText(im, str(i + 1), (px, py), cv.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0), 1)

    # Landmark model location
    PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"
    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()
    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    # Read image
    imageFilename = "./data/girl.jpg"
    im = cv.imread(imageFilename)
    # landmarks will be stored in .txt file
    landmarksBasename = "output_python"

    # Detect faces in the image
    faceRects = faceDetector(im, 0)
    print("Number of faces detected: ", len(faceRects))

    # List to store landmarks of all detected faces
    landmarksAll = []

    # Loop over all detected face rectangles
    for i in range(0, len(faceRects)):
        newRect = dlib.rectangle(int(faceRects[i].left()), int(faceRects[i].top()),
                                 int(faceRects[i].right()), int(faceRects[i].bottom()))

        # For every face rectangle, run landmarkDetector
        landmarks = landmarkDetector(im, newRect)
        # Print number of landmarks
        if i == 0:
            print("Number of landmarks", len(landmarks.parts()))

        # Store landmarks for current face
        landmarksAll.append(landmarks)
        # Draw landmarks on face
        drawLandmarks(im, landmarks)

        landmarksFileName = landmarksBasename + "_" + str(i) + ".txt"
        print("Saving landmarks to", landmarksFileName)
        # Write landmarks to disk
        writeLandmarksToFile(landmarks, landmarksFileName)

    outputFileName = "result_python_Landmarks.jpg"
    print("Saving output image to", outputFileName)
    cv.imwrite(outputFileName, im)

    cv.imshow("Facial Landmark detector", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    #day1()
    #day1_1()
    #day1_2("draw")
    #day3()
    #day3_getAffine()
    #day3_homography()
    #day4("lena")
    #day4_smile()
    #day4_dnn_face("hillary_clinton")
    #day5OR_HOGSVM()
    #day5_DL()
    #day6_YOLO3("./data/bird.jpg", True)
    day7_FaceLandmark()


