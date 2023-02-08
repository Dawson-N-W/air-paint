import cv2
import numpy as np


def getRange(hsv):
    maxValue = 255
    maxHValue = 180
    lowH = 0
    lowS = 0
    lowV = 0
    maxH = maxHValue
    maxS = maxValue
    maxV = maxValue
    
    cv2.namedWindow("Values", 0)
    cv2.resizeWindow("Values", 600, 150)

    values = np.zeros([512, 512, 3], dtype=np.uint8)
    values.fill(255)
    values = cv2.resize(values, (600, 100))

    cv2.createTrackbar("Low Hue", "Values", lowH, maxH, lambda x: x)
    cv2.createTrackbar("Low Saturation", "Values", lowS, maxS, lambda x: x)
    cv2.createTrackbar("Low Value", "Values", lowV, maxV, lambda x: x)
    cv2.createTrackbar("High Hue", "Values", lowH, maxH, lambda x: x)
    cv2.createTrackbar("High Saturation", "Values", lowS, maxS, lambda x: x)
    cv2.createTrackbar("High Value", "Values", lowV, maxV, lambda x: x)

    LHpos = cv2.getTrackbarPos("Low Hue", "Values")
    LSpos = cv2.getTrackbarPos("Low Saturation", "Values")
    LVpos = cv2.getTrackbarPos("Low Value", "Values")
    HHpos = cv2.getTrackbarPos("High Hue", "Values")
    HSpos = cv2.getTrackbarPos("High Saturation", "Values")
    HVpos = cv2.getTrackbarPos("High Value", "Values")

    cv2.putText(values, str(LHpos), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(values, "Low Hue", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(values, str(LSpos), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(values, "Low Sat", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(values, str(LVpos), (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(values, "Low Val", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(values, str(HHpos), (260, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(values, "High Hue", (260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(values, str(HSpos), (340, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(values, "High Sat", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(values, str(HVpos), (420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(values, "High Val", (420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Values", values)

    lowerb = np.array([LHpos, LSpos, LVpos])
    upperb = np.array([HHpos, HSpos, HVpos])

    mymask = cv2.inRange(hsv, lowerb, upperb)

    return mymask


myColors = {'yellow': np.array([[18, 103, 108], [48, 255, 197]]),
            'red': np.array([[168, 134, 41], [180, 255, 255]]),
            'blue': np.array([[110, 112, 23], [147, 217, 255]])}


def findMask(hsv):
    mask = getRange(hsv)
    cv2.imshow("Mask", mask)


def createMask(colorName, frame, hsv):
    mask = cv2.inRange(hsv, myColors[str(colorName)][0], myColors[str(colorName)][1])
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result


def findObject(result, frame, drawContours=False, drawBoundingBox=False, circle=False, color=None):
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(result, (5, 5), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if drawContours:
        cv2.drawContours(canny, contours, -1, (0, 0, 255), 2)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            if drawBoundingBox:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if circle:
                circles.append([int(x + w // 2), int(y), color])  # append the points and the color to the circle list
    return frame


circles = []


def draw(frame):
    for circle in circles:
        cv2.circle(frame, (circle[0], circle[1]), 10, circle[2], -1)  # draw circle at (x,y) ie circle[0], circle[1]
        # and the color tuple at circle[2]


def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (680, 480))
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        redResult = createMask('red', frame, hsv)
        red = findObject(redResult, frame, False, False, True, (0, 0, 255))
        draw(red)

        blueResult = createMask('blue', frame, hsv)
        blue = findObject(blueResult, frame, False, False, True, (255, 0, 0))
        draw(blue)

        # findMask(hsv)

        cv2.imshow("Window", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
