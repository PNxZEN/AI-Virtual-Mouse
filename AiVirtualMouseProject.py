# This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

import cv2
import numpy as np
import HandTrackingModule as htm
import time
from pyautogui import size
from pynput.mouse import Button, Controller as MController
from pynput.keyboard import Key, Controller as KController

##########################
wCam, hCam = 640, 480
frameR = 150  # Frame Reduction
smoothening = 6
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Try primary camera first, fall back to secondary
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# Verify camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set resolution if supported
cap.set(3, wCam)
cap.set(4, hCam)

# Verify actual resolution
actual_width = cap.get(3)
actual_height = cap.get(4)
print(f"Camera resolution: {actual_width}x{actual_height}")

detector = htm.handDetector(maxHands=1)
detector.palm_not_facing_time = 0
wScr, hScr = size()
# print(wScr, hScr)

# Initialize the mouse controller
mouse = MController()
mouse.is_pressed = False

# Initialize the keyboard controller
k = KController()
k.is_pressed = False

while True:
    # Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)


    # Check if the hand is detected
    if len(lmList) != 0 and not detector.isHandSideways():

        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # Only move the mouse if the index finger is up (No need check the other fingers)
        if fingers[1] == 1:
            
            ## Switching windows
            if detector.palm_facing()[2] <= -10000:
                # We need to atleast confirm that the palm is not facing the camera for 1 seconds
                detector.palm_not_facing_time = time.time() if detector.palm_not_facing_time == 0 else detector.palm_not_facing_time
                if time.time() - detector.palm_not_facing_time > 0.2:
                    # If the palm is not facing the camera, we can press alt+tab to show the window switcher
                    detector.palm_not_facing_time = 0
                    if not k.is_pressed:
                        k.press(Key.alt_l)
                        k.tap(Key.tab)
                        k.is_pressed = True                    
            else:
                # If the palm is facing the camera, we need to release the alt key, and click on the window
                if k.is_pressed:
                    mouse.click(Button.left)
                    k.release(Key.alt_l)
                    k.is_pressed = False
            
            ## Moving the mouse
            # Draw a rectangular frame to represent the screen
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

            # Get the center of the hand to move the mouse
            cx, cy = detector.center_of_palm()
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            # Convert coordinates
            x3 = np.interp(cx, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(cy, (frameR, hCam - frameR), (0, hScr))

            # Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move the mouse
            mouse.position = (wScr - clocX, clocY)

            ## Clicking the mouse
            # Check if the thumb tip and index tip are touch
            thumb_tip = lmList[4][1], lmList[4][2]
            index_tip = lmList[8][1], lmList[8][2]
            distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

            # Draw circles around the tips
            cv2.circle(img, thumb_tip, 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, index_tip, 15, (0, 255, 0), cv2.FILLED)
            
            if distance < 30:
                # Draw a circle in red at the midpoint
                cv2.circle(img, (int((thumb_tip[0] + index_tip[0]) / 2), int((thumb_tip[1] + index_tip[1]) / 2)), 15, (0, 0, 255), cv2.FILLED)
                if not mouse.is_pressed and not k.is_pressed:
                    mouse.press(Button.left)
                    mouse.is_pressed = True

            else:
                if mouse.is_pressed:
                    mouse.release(Button.left)
                    mouse.is_pressed = False

        # ## Scrolling (Not working as expected)
        # if fingers[0] == 1 and fingers[1] == 0:
        #     # Calculate the relative movement after normalizing the coordinates
        #     # Calculate scroll distance with an adjustable speed factor
        #     if not hasattr(mouse, 'scroll_start_pos'):
        #         mouse.scroll_start_pos = clocY
            
        #     # Calculate distance from starting position
        #     distance_from_start = clocY - mouse.scroll_start_pos
            
        #     # Reduced base factor and power for slower scrolling
        #     speed_factor = (abs(distance_from_start) / 200) ** 1.2
        #     speed_factor = min(speed_factor, 1.2)  # Limit speed factor to 1.2
            
        #     # Calculate scroll amount
        #     scroll_amount = np.sign(distance_from_start) * speed_factor
            
        #     # Apply smoothing
        #     if not hasattr(mouse, 'last_scroll_amount'):
        #         mouse.last_scroll_amount = 0
            
        #     smoothing_factor = 0.5  # Increased smoothing (reduced from 1 to 0.5)
        #     scroll_amount = (smoothing_factor * scroll_amount + 
        #             (1 - smoothing_factor) * mouse.last_scroll_amount)
            
        #     if abs(scroll_amount) > 0.1:  # Threshold represents minimum scroll distance
        #         mouse.scroll(0, -int(scroll_amount))
        #         mouse.last_scroll_amount = scroll_amount
            
        # Update the previous location
        plocX, plocY = clocX, clocY    

    # Flip the image before displaying
    img = cv2.flip(img, 1)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)