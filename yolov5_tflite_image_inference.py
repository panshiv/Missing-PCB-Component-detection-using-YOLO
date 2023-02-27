from yolov5_tflite_inference import yolov5_tflite
import argparse
import cv2
import time
from PIL import Image
import numpy as np
from utils import letterbox_image, scale_coords


def sortFunc(a):
    store = ""
    # for k, v in value.items():
    #     print(k,v[0][0], v[1])
    # result = {}
    # for d in value:
    #     result.update(d)
    # sorted_values = sorted(result.items(), key=lambda x: x[0])
    # print(sorted_values)
    # for i in sorted_values:
    #     store += i[1]
    # print(store)
    a.sort(key=lambda y: y[0], reverse=True)    # sort by y
    a.sort(key=lambda x: x[1])                  # sort by x
    # print(a)
    first_line = a[:4]
    first_line.sort(key=lambda y: y[0])
    second_line = a[4:8]
    second_line.sort(key=lambda y: y[0])
    third_line = a[8:]
    third_line.sort(key=lambda y: y[0])
    # print("first_line", first_line)
    # print("second_line", second_line)
    # print("third_line", third_line)
    sorted_a = first_line + second_line + third_line                # sort by x
    # print(a)
    for i in range(len(a)):
        store += sorted_a[i][3]
    print(store)
    return store


def detect_image(weights,  image_url, img_size, conf_thres, iou_thres,labels):
    result = []
    start_time = time.time()
    # image = cv2.imread(image_url)
    image = Image.open(image_url)
    original_size = image.size[:2]
    size = (img_size, img_size)
    image_resized = letterbox_image(image, size)
    img = np.asarray(image)

    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image_resized)

    normalized_image_array = image_array.astype(np.float32) / 255.0

    yolov5_tflite_obj = yolov5_tflite(
        weights, labels, img_size, conf_thres, iou_thres)

    result_boxes, result_scores, result_class_names = yolov5_tflite_obj.detect(
        normalized_image_array)

    if len(result_boxes) > 0:
        result_boxes = scale_coords(size, np.array(
            result_boxes), (original_size[1], original_size[0]))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (20, 40)

        # fontScale
        fontScale = 0.5

        # Blue color in RGB
        color = (0, 0, 255)

        # Line thickness of 1 px
        thickness = 1
        flag = 0
        last = []
        for i, r in enumerate(result_boxes):
            if(flag == 0):

                org = (int(r[0]), int(r[1]))
                # labelImg = img[int(r[1]):int(r[3]), int(r[0])
                #                    : int(r[2]), :].copy()

                cv2.rectangle(img, (int(r[0]), int(r[1])),
                              (int(r[2]), int(r[3])), (255, 0, 0), 3)
                cv2.putText(img, str(result_class_names[i]), (org[0], org[1]+15), font,
                            fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(img, str(int(100*result_scores[i])) + '% ', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                last.append(i)
                # result.append(
                #     {round(r[0], 2): result_class_names[i]})
                result.append(
                    [round(r[0], 2), round(r[1], 2), int(100*result_scores[i]), result_class_names[i]])

                # cv2.imshow("frame2", labelImg[:, :, ::-1])
                # print("detect character")
                # cv2.imshow("frame", img[:, :, ::-1])

                # print("in flag", flag)
        # break
        flag = 1
        save_result_filepath = image_url.split(
            '/')[-1].split('.')[0] + 'yolov5_output.jpg'
        # save_cropped_result_filepath = image_url.split(
        #     '/')[-1].split('.')[0] + 'cropped_yolov5_output.jpg'
        cv2.imwrite("./output/images/"+save_result_filepath, img[:, :, ::-1])
        # cv2.imwrite("output/cropped/"+save_cropped_result_filepath,
        #             labelImg[:, :, ::-1])

        end_time = time.time()

        print('FPS:', 1/(end_time-start_time))
        print('Total Time Taken:', end_time-start_time)
        # print(result)

        value = sortFunc(result)
        if flag == 1 or cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str,
                        default='./models/best-fp16.tflite', help='model.tflite path(s)')
    parser.add_argument('-i', '--img_path', type=str,
                        required=True, help='image path')
    parser.add_argument('--img_size', type=int, default=640, help='image size')
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--labels', type=str,
                        default="./labels/classes.txt", help='label path')

    opt = parser.parse_args()

    # print(opt)
    detect_image(opt.weights, opt.img_path, opt.img_size,opt.conf_thres, opt.iou_thres,opt.labels)
