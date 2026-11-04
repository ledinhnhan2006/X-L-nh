import cv2
import matplotlib.pyplot as plt

img = cv2.imread("room.jpg")

if img is None:
    print("Khong tim thay anh")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

plt.imshow(blur, cmap="gray")
plt.axis("off")
plt.show()