import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
 
# Creating a video path and opening the video file
CURRENT_DIR = os.path.dirname(__file__)
video_path = os.path.join(CURRENT_DIR,'ball.mov')
ball_video = cv2.VideoCapture(video_path)

# Creating a list of the coordinates of the center of the ball
center_x_coordinates = []
center_y_coordinates = []

# Displaying an error message if there is trouble opening the video file
if (ball_video.isOpened() == False):
    print("Error opening the video file")

# Playing the video 
while(ball_video.isOpened()):
    ret, frame = ball_video.read()

    fps = ball_video.get(5)

    if ret == True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 190, 35])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([170, 180, 35])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        result = cv2.bitwise_and(frame, frame, mask=mask)
        y1, x1 = np.nonzero(mask)
        
        ball_pixels = np.nonzero(mask)

        if ball_pixels:
            try :
                x_coordinates = int(np.mean(ball_pixels[1]))
                y_coordinates = int(np.mean(ball_pixels[0]))
            except:
                x_coordinates = 0
                y_coordinates = 0
        
        print(f"Center coordinates : ({x_coordinates},{(y_coordinates)})")

        cv2.circle(frame, (x_coordinates, y_coordinates), 5, (0, 255, 0), -1)

        center_x_coordinates.append(np.mean(x1))
        center_y_coordinates.append(np.mean(y1))
        x2 = np.array(center_x_coordinates)
        y2 = np.array(center_y_coordinates)

        # cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        # cv2.imshow("result", result)

        key = cv2.waitKey(5)
        
        if key == ord('q'):
            break
    else:
        break

# Release the video
ball_video.release()
cv2.destroyAllWindows()

# Plot the coordinates of the center of the ball
plt.plot(center_x_coordinates, center_y_coordinates, 'o', label='data')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Coordinate plot")
plt.show()


# Masking the nan values from the numpy array of center coordinates
mask3 = np.isnan(x2)
x = x2[~mask3]
mask4 = np.isnan(y2)
y = y2[~mask4]

# Finding the equation of the parabola using the least squaresult method
X = np.column_stack([x**2, x, np.ones_like(x)])
Y = np.array(y)
XtX = np.dot(X.T, X)
XtX_inv = np.linalg.inv(XtX)
XtY = np.dot(X.T, Y)
B = np.dot(XtX_inv, XtY)

# Displaying the equation of the parabola
x3 = np.linspace(np.min(x), np.max(x))
y3 = B[0]*x3**2 + B[1]*x3 + B[2]
print(f"The equation of parabola is \ny = {B[0]}x^2 + {B[1]}x + {B[2]}")

# Plotting the resultult
plt.plot(x, y, 'o', label='data')
plt.plot(x3, y3, label='best fit')
plt.legend()
plt.show()


land_x = 0
land_y = y[0] + 300
B[2] -= land_y

# Solving the quadtaric equation to get the solution
discriminant = math.sqrt(B[1]**2 - 4*B[0]*B[2])
land_x1 = (-B[1] + discriminant) / (2*B[0])
land_x2 = (-B[1] - discriminant) / (2*B[0])
land_x = max(land_x1, land_x2)

# Displaying the landing coordinates of the ball
print(f"The ball will land on {land_x,land_y}")
