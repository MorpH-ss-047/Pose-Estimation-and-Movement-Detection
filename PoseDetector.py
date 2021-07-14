import time
import cv2
import mediapipe as mp


class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def getPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getLandMarks(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = PoseDetector()
    lmLists = []
    while True:
        ret, frame = cap.read()
        img = detector.getPose(frame)
        lmList = detector.getLandMarks(frame)
        lmLists.append(lmList)

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv2.putText(frame, "FPS: %.2f" % fps, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('Img', img)
        print(len(lmLists), len(lmList), len(lmLists[-1]))
        print(lmLists)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
