import cv2
import dlib


def initialize_detector(args):
    detector = None
    if args.detect == "opencv_haar":
        detector = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    elif args.detect == "opencv_dnn":
        if args.opencv_dnn_type == "CAFFE":
            model_file = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "./models/deploy.prototxt"
            detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
        else:
            model_file = "./models/opencv_face_detector_uint8.pb"
            config_file = "./models/opencv_face_detector.pbtxt"
            detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    elif args.detect == "dlib_hog":
        detector = dlib.get_frontal_face_detector()
    elif args.detect == "dlib_dnn":
        detector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
    return detector


def detect_face_opencv_haar(frame, face_cascade):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    bboxes = []
    scale_width = 1.0
    scale_height = 1.0
    for x, y, w, h in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cv_rect = [int(x1 * scale_width), int(y1 * scale_height),
                   int(x2 * scale_width), int(y2 * scale_height)]
        bboxes.append(cv_rect)
        # cv2.rectangle(frame, (cv_rect[0], cv_rect[1]), (cv_rect[2], cv_rect[3]), (0, 255, 0))
        # int(round(frameHeight / 150)), 4)
    return bboxes


def detect_face_opencv_dnn(frame, net, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))  # int(round(frame_height/150)), 8)
    return bboxes


def detect_face_dlib_hog(frame, detector):
    scale_height = 1.0
    scale_width = 1.0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_rects = detector(frame_rgb, 0)
    bboxes = []
    for face_rect in face_rects:
        cv_rect = [int(face_rect.left() * scale_width), int(face_rect.top() * scale_height),
                   int(face_rect.right() * scale_width), int(face_rect.bottom() * scale_height)]
        bboxes.append(cv_rect)
        # cv2.rectangle(frame, (cv_rect[0], cv_rect[1]), (cv_rect[2], cv_rect[3]), (0, 255, 0))
        # int(round(frameHeight/150)), 4)
    return bboxes


def detect_face_dlib_mmod(frame, detector):
    scale_height = 1.0
    scale_width = 1.0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_rects = detector(frame_rgb, 0)
    bboxes = []
    for face_rect in face_rects:
        cv_rect = [int(face_rect.rect.left() * scale_width), int(face_rect.rect.top() * scale_height),
                   int(face_rect.rect.right() * scale_width), int(face_rect.rect.bottom() * scale_height)]
        bboxes.append(cv_rect)
        # cv2.rectangle(frame, (cv_rect[0], cv_rect[1]), (cv_rect[2], cv_rect[3]), (0, 255, 0))
        # int(round(frameHeight/150)), 4)
    return bboxes
