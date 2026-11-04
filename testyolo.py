from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("room.jpg")

for result in results:

    for box in result.boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cls = int(box.cls[0])
        confidence = float(box.conf[0])

        name = model.names[cls]

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        print("-------------------")
        print("Object:", name)
        print("Confidence:", confidence)
        print("Position:", center_x, center_y)
        print("Box:", x1, y1, x2, y2)