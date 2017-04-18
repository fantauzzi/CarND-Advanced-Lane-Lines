import cv2
import matplotlib.pyplot as plt


f_name = 'test_images/straight_lines1.jpg'
img = cv2.imread(f_name)
assert img is not None

img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
plt.imshow(img)
plt.show()


