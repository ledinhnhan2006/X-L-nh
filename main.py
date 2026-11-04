import cv2
import matplotlib.pyplot as plt

img = cv2.imread("room.jpg")

if img is None:
    print("Khong tim thay anh")
    exit()

# 1. Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Làm mờ ảnh
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Tìm đường biên
edges = cv2.Canny(blur, 100, 200)

# 4. Tìm contour
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 5. Lọc các vùng có diện tích lớn
areas = []

for contour in contours:
    area = cv2.contourArea(contour)

    if area > 500:
        areas.append(area)

print("So vung lon:", len(areas))

# 6. Vẽ contour
result = img.copy()

cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.imshow(result)
plt.axis("off")
plt.show()