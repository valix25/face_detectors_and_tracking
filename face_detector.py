from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import dlib

from detectors import initialize_detector, detect_face_opencv_haar, detect_face_opencv_dnn, detect_face_dlib_hog
from detectors import detect_face_dlib_mmod


def run_face_detection(frame, face_detector, args):
    bboxs = []
    if args.detect == "opencv_haar":
        bboxs = detect_face_opencv_haar(frame, face_detector)
    elif args.detect == "opencv_dnn":
        bboxs = detect_face_opencv_dnn(frame, face_detector)
    elif args.detect == "dlib_hog":
        bboxs = detect_face_dlib_hog(frame, face_detector)
    elif args.detect == "dlib_dnn":
        bboxs = detect_face_dlib_mmod(frame, face_detector)
    return bboxs


def handle_face_detection_and_tracking(frame, face_detector, args, counter, tracker):
    if counter % args.tracking_freq == 0:
        bboxs = run_face_detection(frame, face_detector, args)
        if len(bboxs) > 0:
            new_tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(bboxs[0][0], bboxs[0][1], bboxs[0][2], bboxs[0][3])
            new_tracker.start_track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rect)
            return new_tracker, bboxs[0]
        else:
            return None, (0, 0, 0, 0)
    else:
        if tracker is not None:
            # update tracker
            tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pos = tracker.get_position()

            # unpack the position object
            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())
            return tracker, (start_x, start_y, end_x, end_y)
        else:
            return None, (0, 0, 0, 0)


def main():
    # 1. Parse command line inputs
    mode_options = ["no_detection", "opencv_haar", "opencv_dnn", "dlib_hog", "dlib_dnn"]
    parser = argparse.ArgumentParser()
    # 'default' is the value that the attribute gets when the argument is absent
    # 'constant' is the value the attribute gets when present
    parser.add_argument("--detect", default=mode_options[0], const=mode_options[1], nargs='?', choices=mode_options,
                        help="Options for face detection: " + str(mode_options))
    parser.add_argument("--tracking", action="store_true", help="If specified, tracking is on")
    parser.add_argument("--resize_width", type=int, default=480, help="Resize input frame. Maintains aspect ratio.")
    parser.add_argument("--opencv_dnn_type", default=["TF"], choices=["TF", "CAFFE"])
    parser.add_argument("--tracking_freq", type=int, default=10, help="Tracking frequency.")
    args = parser.parse_args()

    # 2. Initialize the video stream, allow the camera sensor to warm up, and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    face_detector = initialize_detector(args)
    tracker = None
    counter = 0
    # 3. Process the camera input frame by frame
    while True:
        # 3.1. Read frame and resize
        # height x width x 3
        frame = vs.read()
        if frame is None:
            print("Video stream broke.")
            break
        width = args.resize_width
        height = int(frame.shape[0] * width * 1.0 / frame.shape[1])
        if width < frame.shape[1]:
            # for shrinking use cv2.INTER_AREA
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        else:
            # for upscaling use cv2.INTER_LINEAR or cv2.INTER_CUBIC (slower)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        # 3.2. Run face detection
        output_frame = frame.copy()
        if args.tracking:
            tracker, bbox = handle_face_detection_and_tracking(frame, face_detector, args, counter, tracker)
            if tracker is not None:
                output_frame = cv2.rectangle(output_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0))
                y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
                if counter % args.tracking_freq == 0:
                    cv2.putText(output_frame, "Detecting", (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(output_frame, "Tracking", (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            bboxs = run_face_detection(frame, face_detector, args)
            if len(bboxs) > 0:
                cv_rect = bboxs[0]
                output_frame = cv2.rectangle(output_frame, (cv_rect[0], cv_rect[1]), (cv_rect[2], cv_rect[3]),
                                             (0, 255, 0))
        counter += 1

        # 3.3. Show the output frame
        if output_frame is not None:
            cv2.imshow("Frame", output_frame)
        key = cv2.waitKey(1) & 0xFF

        # 3.4. If the `q` key was pressed, break from the loop
        if key == ord("q"):
            print("Input size: ", height, " x ", width)
            break

        fps.update()

    # 4. Stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # 5. Do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()
