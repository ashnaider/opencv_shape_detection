import os
import cv2
import numpy as np


INPUT_VIDEO_PATH = os.path.join("input", "shapes_video.mp4")
OUTPUT_DIR = "output"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "detected_shapres.mp4")

blue  = (255, 0, 0)
green = (0, 255, 0)
red   = (0, 0, 255)
black = (0, 0, 0)


def write_shape(image, shape, text, position):
    offset = 0
    color = black

    if text == "triangle":
        offset = -50
        color = blue
        
    elif text == "circle":
        offset = -30
        color = red

    elif text == "rectangle":
        offset = -50
        color = green

    x, y = position
    position = (x + offset, y)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_stroke = 3

    cv2.drawContours(image, [shape], -1, color, thickness=4)
    cv2.putText(image, text, position, font, font_size, color, font_stroke) 


def with_in_eps(a, b, eps=4) -> bool:
    return abs(a - b) < eps


def is_on_border(image, bb_coord):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bb_coord
    eps = 1

    a = with_in_eps(x1, 0, eps)
    b = with_in_eps(x1+x2, w, eps)
    c = with_in_eps(y1, 0, eps)
    d = with_in_eps(y1+y2, h, eps)

    return any([a, b, c, d])


def is_square(bb_coord):
    """Check if shape is square"""
    _, _, w, h = bb_coord
    
    w, h = max(w, h), min(w, h)
    return (w - h) * 100 / w < 20  # if width bigger than height only by 20% 


def main():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not os.path.exists(OUTPUT_DIR) or not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width,frame_height))

    kernel_3 = np.ones((3,3), np.uint8)
    kernel_5 = np.ones((5,5), np.uint8)

    w, h = 7, 2
    kernel_h = np.ones((h, w), np.uint8)
    kernel_w = np.ones((w, h), np.uint8)

    if cap.isOpened() == False:
        print("Error File Not Found")


    while cap.isOpened():
        ret,frame= cap.read()

        if ret == False:
            break

        tmp = frame
        tmp = cv2.blur(tmp, (7, 7))

        hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
        lower_range = np.array([50,15,25])
        upper_range = np.array([90,190,220])
        mask = cv2.inRange(hsv, lower_range, upper_range)

        tmp = mask

        tmp = cv2.dilate(tmp, kernel_3, iterations=4)

        tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel_5, iterations=3)

        contours, hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        shapes = []
        for contour in contours:
            if len(contour) > 70:  
                shapes.append(contour)


        hulles = [cv2.convexHull(shape) for shape in shapes]        

        for shape in hulles:
            approx = cv2.approxPolyDP(shape, 0.03*cv2.arcLength(shape, True), closed=True)
            
            corners = len(approx)

            br = cv2.boundingRect(shape)
            M = cv2.moments(shape)

            if not is_on_border(frame, br):

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                position = (cx, cy)

                if corners == 3:
                    write_shape(frame, shape, "triangle", position)

                elif corners == 4:
                    write_shape(frame, shape, "rectangle", position)

                elif corners >= 6:
                    if is_square(br):
                        write_shape(frame, shape, "circle", position)


        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()