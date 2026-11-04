
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def calculate_score(image):
    height, width = image.shape[:2]
    total_pixels = height * width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 80, 160)

    edge_density = np.count_nonzero(edges) / total_pixels

    edge_penalty = min(edge_density * 300, 100)

    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.threshold(
        edges, 50, 255, cv2.THRESH_BINARY
    )[1]

    binary = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    large_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > total_pixels * 0.001:
            large_contours.append(contour)

    object_count = len(large_contours)

    object_penalty = min(object_count * 3, 100)

    occupied_area = sum(
        cv2.contourArea(c) for c in large_contours
    )

    area_ratio = occupied_area / total_pixels

    area_penalty = min(area_ratio * 200, 100)

    lower_region = gray[int(height * 0.5):, :]

    lower_edges = cv2.Canny(
        lower_region, 80, 160
    )

    lower_density = (
        np.count_nonzero(lower_edges)
        / lower_edges.size
    )

    clutter_penalty = min(lower_density * 300, 100)

    score = 100 - (
        0.25 * edge_penalty
        + 0.30 * object_penalty
        + 0.25 * clutter_penalty
        + 0.20 * area_penalty
    )

    score = max(0, min(100, score))

    return (
        round(score, 2),
        object_count,
        edge_density,
        area_ratio,
        clutter_penalty,
        large_contours
    )


def classify_room(score):
    if score >= 80:
        return "Phong gon gang"

    elif score >= 60:
        return "Phong kha gon"

    elif score >= 40:
        return "Phong hoi bua bon"

    else:
        return "Phong bua bon"


def main():
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Chon anh phong",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if not image_path:
        print("Chua chon anh")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("Khong doc duoc anh")
        return

    (
        score,
        object_count,
        edge_density,
        area_ratio,
        clutter_penalty,
        contours
    ) = calculate_score(image)

    label = classify_room(score)

    result = image.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    text = f"Score: {score}/100"

    cv2.putText(
        result,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    print("\n===== KET QUA CHAM DIEM =====")
    print("Anh:", os.path.basename(image_path))
    print("Diem:", score, "/100")
    print("Danh gia:", label)
    print("So vung phat hien:", object_count)
    print("Mat do bien:", round(edge_density, 4))
    print("Ty le dien tich vung:", round(area_ratio, 4))
    print("Chi so roi o nua duoi:", round(clutter_penalty, 2))

    os.makedirs("results", exist_ok=True)

    output_path = os.path.join(
        "results",
        "result_" + os.path.basename(image_path)
    )

    cv2.imwrite(output_path, result)

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(12, 7))
    plt.imshow(result_rgb)
    plt.title(f"{label} - {score}/100")
    plt.axis("off")
    plt.show()

    print("Da luu anh ket qua:", output_path)


if __name__ == "__main__":
    main()
