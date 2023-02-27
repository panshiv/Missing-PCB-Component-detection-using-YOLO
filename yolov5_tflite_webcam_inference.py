from yolov5_tflite_inference import yolov5_tflite
import argparse
import cv2
from PIL import Image
from utils import letterbox_image, scale_coords
import numpy as np
import time
from yolov5_tflite_image_inference import detect_image
import os


def detect_video(weights, labels, webcam, img_size, conf_thres, iou_thres):
    BASE_PATH = os.getcwd()

    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    video = cv2.VideoCapture(webcam)
    fps = video.get(cv2.CAP_PROP_FPS)
    # print(fps)
    h = int(video.get(3))
    w = int(video.get(4))
    print(w, h)
    # h = 1280
    # w = 720
    result_video_filepath = 'webcam_yolov5_output.avi'
    out = cv2.VideoWriter(result_video_filepath, fourcc, int(fps), (h, w))

    yolov5_tflite_obj = yolov5_tflite(
        weights, labels, img_size, conf_thres, iou_thres)
    size = (img_size, img_size)
    try:
        while True:

            check, frame = video.read()

            if not check:
                break
            # frame = cv2.resize(frame,(h,w))
            # no_of_frames += 1
            image_resized = letterbox_image(Image.fromarray(frame), size)
            image_array = np.asarray(image_resized)

            normalized_image_array = image_array.astype(np.float32) / 255.0
            result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(
                normalized_image_array)
            if len(result_boxes) > 0:
                result_boxes = scale_coords(
                    size, np.array(result_boxes), (w, h))

            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (20, 40)

            # fontScale
            fontScale = 0.5

            # Blue color in BGR
            color = (0, 255, 0)

            # Line thickness of 1 px
            thickness = 1
            flag = 0
            for i, r in enumerate(result_boxes):
                if (flag == 0 and result_scores[i] > 0.4):

                    org = (int(r[0]), int(r[1]))
                    labelImg = frame[int(r[1]):int(
                        r[3]), int(r[0]): int(r[2]), :].copy()

                    cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(
                        r[2]), int(r[3])), (255, 0, 0), 1)
                    cv2.putText(frame, str(int(100*result_scores[i])) + '%  ' + str(result_class_names[i]), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

                    cv2.imshow("frame2", labelImg)
                    cv2.imwrite("./output/cropped/cropped1.jpg", labelImg)
                    flag = 1
                    print(flag)
                    break

            # save_cropped_result_filepath = image_url.split(
            # '/')[-1].split('.')[0] + 'cropped_yolov5_output.jpg'
            out.write(frame)

            # uncomment below lines to see the output
            cv2.imshow('output', frame)
            if (flag == 1):
                value = detect_image(weights=BASE_PATH+"/models/best-fp16.tflite", labels=BASE_PATH+"/labels/classes.txt", conf_thres=0.25, iou_thres=0.45,
                                     image_url=BASE_PATH+"/output/cropped/cropped1.jpg", img_size=640)
                print("Inside cam", value)
                return value
            if (cv2.waitKey(1) & 0xFF == ord('q') or flag == 1):
                break
            end_time = time.time()
            print('FPS:', 1/(end_time-start_time))
            start_time = end_time

        out.release()

    except:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str,
                        default='./models/best-fp16.tflite', help='model.tflite path(s)')
    parser.add_argument('-wc', '--webcam', type=int,
                        default=0, help='webcam number 0,1,2 etc.')
    parser.add_argument('--img_size', type=int, default=640, help='image size')
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--labels', type=str,
                        default="./labels/classes.txt", help='label path')
    opt = parser.parse_args()

    detect_video(opt.weights, opt.webcam, opt.img_size,
                 opt.conf_thres, opt.iou_thres, opt.labels)
