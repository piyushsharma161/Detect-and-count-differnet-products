import cv2
import argparse
import orien_lines
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import pandas as pd
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy 
import numpy as np
from flask import Response
from flask import Flask
from flask import render_template
import threading

app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')

args = vars(ap.parse_args())



args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(predict(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def predict():
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.87
    
    #vs = cv2.VideoCapture('rtsp://192.168.1.64')
    vs = VideoStream(0).start()
    #Oriendtation of machine    
    Orientation= 'bt'


    # max number of products we want to detect/track
    num_products_detect = 50

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draw bounding boxeses and text

            detector_utils.draw_box_on_image(
                num_products_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame,
                Orientation)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:

                product = []
                count_product = []
                for i in range(boxes.shape[0]):
                    if scores[i] > score_thresh:
                        product_name = detector_utils.category_index[classes[i]]['name']
                        if product_name not in product:
                            product.append(product_name)
                            count_product.append(1)
                        else:
                            index = product.index(product_name)
                            count_product[index] = count_product[index] + 1

                x = 20
                origin2 = 20
                for i in range(len(product)):
                    cv2.putText(frame, str(product[i]) + ' : ' + str(count_product[i]), (0, origin2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 255), 2)
                    origin2 = origin2 + x

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str("{0:.2f}".format(fps)), (550, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                                                                              0.5, (77, 255, 9), 1)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    vs.stop()
                    break
        

        print("Average FPS: ", str("{0:.2f}".format(fps)))
        
    except KeyboardInterrupt:
        today = date.today()
        print("Average FPS: ", str("{0:.2f}".format(fps)))

# check to see if this is the main thread of execution
if __name__ == '__main__':
    app.run(debug=True)

'''
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False) '''
