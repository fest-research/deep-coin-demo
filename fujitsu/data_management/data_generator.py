import cv2
import numpy as np

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    i = 0
    counter = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                counter += 1
                print(counter)
                # crop the frame to something smaller
                half_size = 100
                center_y = frame.shape[0] // 2
                center_x = frame.shape[1] // 2
                frame = frame[center_y - half_size:center_y + half_size,
                              center_x - half_size:center_x + half_size]
                cv2.imshow('video', frame)

                if counter % 20 == 0:
                    cv2.imwrite('data/image{}.jpg'.format(i), frame)
                    i+=1
                    counter = 0

        if cv2.waitKey(10) == 27:
            break
