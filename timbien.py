import cv2
import matplotlib.pyplot as plt

img = cv2.imread("room.jpg")

if img is None:
    print("Khong tim thay anh")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 100, 200)
edge_pixels = cv2.countNonZero(edges)

total_pixels = edges.shape[0] * edges.shape[1]

edge_density = edge_pixels / total_pixels

print("Edge density:", edge_density)

plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()