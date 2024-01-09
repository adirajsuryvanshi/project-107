import cv2
import time
import math
import numpy as np

# Define the parameters for the object tracker
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Define the goal position
p1 = 530
p2 = 300

# Define arrays to store the x and y coordinates of the tracked objects
xs = []
ys = []

# Load the video
video = cv2.VideoCapture("footvolleyball.mp4")

# Load the cascade file for the goal detection
cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Read the first frame of the video
success, img = video.read()

# Select the bounding box for the goal
bbox = cv2.selectROI("tracking", img, False)

# Initialize the object tracker with the selected bounding box
tracker = cv2.TrackerCSRT_create()
tracker.init(img, bbox)

# Define the goal_track() function to draw the goal and track the ball
def goal_track(img, bbox):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper boundaries of the yellow color in HSV
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find the contours of the yellow objects
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle around the goal
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Find the largest contour and draw it on the image
    goal = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [goal], -1, (0, 255, 255), 2)

    # Find the center of the goal and draw a circle
    M = cv2.moments(goal)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

    # Calculate the distance between the ball and the goal
    dist = math.sqrt(((cX - p1)**2) + (cY - p2)**2)

    # If the ball is within 20 pixels of the goal, draw a "Goal" text
    if dist <= 20:
        cv2.putText(img, "Goal", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Append the x and y coordinates of the tracked objects to the arrays
    xs.append(cX)
    ys.append(cY)

    # Loop through the tracked objects and draw circles
    for i in range(len(xs) - 1):
        cv2.circle(img, (xs[i], ys[i]), 2, (0, 0, 255), 5)

    # Return the image with the goal and ball drawn
    return img

# Define the drawBox() function to draw the bounding box for the goal
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Start the video loop
while True:
    # Read a frame from the video
    success, img = video.read()

    # Update the object tracker with the current frame
    success, bbox = tracker.update(img)

    # Perform the goal detection and tracking
    goal_track(img, bbox)

    # Perform the goal detection using the cascade classifier
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    goals = cascade.detectMultiScale(gray, 1.3, 5)

    # Draw the bounding boxes for the goals
    for (x, y, w, h) in goals:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("tracking", img)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("closing")
        break

# Release the video and close the window
video.release()
cv2.destroyAllWindows()