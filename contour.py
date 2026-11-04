import cv2
import matplotlib.pyplot as plt

img = cv2.imread("room.jpg")

if img is None:
    print("Khong tim thay anh")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 100, 200)

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = img.copy()

cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
areas = []

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 500:
        areas.append(area)

print("So vung lon:", len(areas))
plt.imshow(result)
plt.axis("off")
plt.show()

print("So contour:", len(contours))