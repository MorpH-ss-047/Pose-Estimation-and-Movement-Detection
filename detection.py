import time
import cv2
import PoseDetector as pdr

# path to video file
path = 'vids/2.mp4'
# pass 0 if you want to detect from webcam
cap = cv2.VideoCapture(path)
detector = pdr.PoseDetector()
ptime = 0
lmLists = []


def is_motion():
    lm = lmLists[-1]
    lm2 = lmLists[-2]
    if len(lm) > 0 and len(lm2) > 0:
        for i, j in zip(lm, lm2):
            for _ in range(len(i)):
                if abs(i[1] - j[1]) > 15 or abs(i[2] - j[2]) > 15:
                    return True
        return False


def detect_motion(frame):
    if len(lmLists) > 1:
        if not is_motion():
            cv2.putText(frame, 'No motion', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Motion detected', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


while(True):
    ret, frame = cap.read()
    frame = detector.getPose(frame)
    lmList = detector.getLandMarks(frame, draw=False)
    lmLists.append(lmList)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(frame, 'FPS: %.2f' % fps, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    detect_motion(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
