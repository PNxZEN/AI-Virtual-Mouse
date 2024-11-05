import cv2
print("Imported cv2")
import mediapipe as mp
print("Imported mp")
import time
import math
print("Imported math, time")
import numpy as np
print("Imported numpy")

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def center_of_palm(self):
        # Get the center of the palm
        if not self.lmList:
            return None

        # Get the points 5,17,0
        imcp = self.lmList[5][1], self.lmList[5][2]
        pmcp = self.lmList[17][1], self.lmList[17][2]
        wrist = self.lmList[0][1], self.lmList[0][2]

        # Calculate the center of the palm
        cx = (imcp[0] + pmcp[0] + wrist[0]) // 3
        cy = (imcp[1] + pmcp[1] + wrist[1]) // 3

        return cx, cy
    
    def palm_facing(self, threshold=2):
        # Get key points
        index_mcp = self.lmList[5][1], self.lmList[5][2]
        pinky_mcp = self.lmList[17][1], self.lmList[17][2]
        wrist = self.lmList[0][1], self.lmList[0][2]

        # Check if all points are detected
        if None in index_mcp or None in pinky_mcp or None in wrist:
            return None
        
        # Check for left hand or right hand
        handedness = self.results.multi_handedness[0].classification[0].label
        
        # Calculate vectors and add z=0 coordinate
        v1 = np.array([*pinky_mcp, 0]) - np.array([*wrist, 0])
        v2 = np.array([*index_mcp, 0]) - np.array([*wrist, 0])
        
        # Calculate cross product
        cross_product = np.cross(v1, v2)
        
        # For right hand, if z-component is positive, palm is facing camera
        if handedness == "Right":
            cross_product != -cross_product

        return cross_product
    
    def is_palm_facing_camera(self, threshold=0):
        # Palm facing camera when z-component is positive
        return self.palm_facing()[2] > threshold*1000
        
    def isHandSideways(self):
        z = self.palm_facing()[2]
        return -3000 < z < 3000
    
    def fingersUp(self, img=None):
        if not self.lmList or self.isHandSideways():
            return [None, None, None, None, None]
        
        fingers = []

        # Make a circle by taking a center at the palm center and radius should be dynamically calculated
        cx, cy = self.center_of_palm()
        cy -= 30  # Adjust the center of the circle to be slightly above the palm center
        
        # Radius should be roughly 1.2 times the distance between the palm center and the index finger mcp 
        radius = int(1.5 * math.hypot(self.lmList[5][1] - cx, self.lmList[5][2] - cy))
        if img is not None: cv2.circle(img, (cx, cy), radius, (0, 255, 0), cv2.FILLED)
        
        ## Thumb
        # If the Thumb mcp or Thumb tip is inside the circle, it is closed
        thumb_tip = self.lmList[4][1], self.lmList[4][2]
        distance_tip = math.hypot(thumb_tip[0] - cx, thumb_tip[1] - cy)

        if distance_tip < radius:
            fingers.append(0)
        else:
            fingers.append(1)


        ## Other fingers
        for id in range(1, 5):
            # Get tip coordinates
            tip_x, tip_y = self.lmList[self.tipIds[id]][1], self.lmList[self.tipIds[id]][2]
            
            # Calculate Euclidean distance from tip to the center of the circle
            distance = math.hypot(tip_x - cx, tip_y - cy)
            
            # If the finger coincides with the circle, it is closed
            if distance > radius:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Flip first before detection
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        
        if len(lmList) != 0:
            fingersUp = detector.fingersUp(img)
            # print(fingersUp, end="\r")
            palm_facing = detector.is_palm_facing_camera(2)
            cv2.putText(img, f"Palm Facing: {palm_facing}", 
                       (10, 120), cv2.FONT_HERSHEY_PLAIN, 2,
                       (0, 255, 0) if palm_facing else (0, 0, 255), 2)
            
            # Draw the angle of the palm
            angle = detector.palm_facing()[2]
            cv2.putText(img, f"Angle: {angle}", (10, 150), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            print(f"Angle: {angle}") if angle < -50000 else None
            
            # Draw the center of the palm
            cx, cy = detector.center_of_palm()
            # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()