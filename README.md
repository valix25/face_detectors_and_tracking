Face detectors and tracking

Project based on the following articles (entirely recommend for anyone generally interested in the topics):

- Face Detection – OpenCV, Dlib and Deep Learning ( C++ / Python ) 
  https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
  
- Object tracking with dlib
  https://www.pyimagesearch.com/2018/10/22/object-tracking-with-dlib/
  
To run the current code (assuming you have installed dlib and opencv for python3.6, recommend to use a virtual environment):

usage: face_detector.py [-h]

[--detect [{no_detection,opencv_haar,opencv_dnn,dlib_hog,dlib_dnn}]]

[--tracking] [--resize_width RESIZE_WIDTH]

[--opencv_dnn_type {TF,CAFFE}]

[--tracking_freq TRACKING_FREQ]

optional arguments:

  -h, --help            show this help message and exit
  
  --detect [{no_detection,opencv_haar,opencv_dnn,dlib_hog,dlib_dnn}]
                        Options for face detection: ['no_detection',
                        'opencv_haar', 'opencv_dnn', 'dlib_hog', 'dlib_dnn']

  --tracking            If specified, tracking is on
  
  --resize_width RESIZE_WIDTH
                        Resize input frame. Maintains aspect ratio.
                        
  --opencv_dnn_type {TF,CAFFE}
  
  --tracking_freq TRACKING_FREQ
                        Tracking frequency.
