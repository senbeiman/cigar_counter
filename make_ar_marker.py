import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
generator = cv2.aruco.drawMarker(dictionary, 1, 100)
cv2.imwrite('1.png', generator)
generator = cv2.aruco.drawMarker(dictionary, 2, 100)
cv2.imwrite('2.png', generator)
