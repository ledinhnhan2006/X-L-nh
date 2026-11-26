# Góc Gọn — Chấm điểm ảnh phòng với YOLO

**YOLO segmentation phiên bản 1:** dataset 140 ảnh trên Roboflow, checkpoint được chọn đã fine-tune từ COCO bằng Ultralytics, đủ 12 lớp. Gói app kèm checkpoint dùng `start_segmentation_windows.bat`. Xem [mô hình, cách mở và kết quả kiểm tra](MODEL_SEGMENTATION_V1.md).

Mở bản segmentation bằng `start_segmentation_windows.bat`; gói app tải về đã kèm `weights_segmentation.pt`. Launcher tự gọi trình cài khi chưa có môi trường Python. App hỗ trợ thêm ảnh, khoanh vùng, quy tắc vị trí, mức phạt theo lớp và diện tích, tải ảnh kết quả và báo cáo JSON.

Nếu chỉ tải mã nguồn GitHub, đặt checkpoint segmentation đã huấn luyện dưới tên `weights_segmentation.pt` cạnh `web_app.py` trước khi mở launcher segmentation. Gói app đầy đủ được cung cấp riêng cùng dataset và checkpoint.

**Hướng dẫn sử dụng và công thức v4 — diện tích contour, điểm nền 100, sàn 40:** [PHUONG_PHAP_CHAM_DIEM.md](PHUONG_PHAP_CHAM_DIEM.md).

Checkpoint detection cũ `weights.pt` có 11 lớp, thiếu backpack; xem [MODEL_CARD.md](MODEL_CARD.md). Các launcher `start_windows.bat`/`start_app.bat` dùng lựa chọn này khi không đặt YOLO_MODEL_PATH. Checkpoint segmentation mới có 12 lớp, được chọn bằng launcher segmentation. Đây là điểm thử nghiệm; không có điểm nếu không đánh giá được vật thể.

Diện tích lấy từ mask → contour → cv2.contourArea khi dùng segmentation; mô hình detection dùng GrabCut để ước lượng contour. App hiển thị đường bao và trường hợp chưa đo được. Xem PHUONG_PHAP_CHAM_DIEM.md để biết giới hạn.

**Chuẩn bị dữ liệu instance segmentation:** [bộ nháp 130 ảnh (100 cũ + 30 mới)](training/segmentation_v1/README.md), có polygon nguồn và mặt nạ SAM hỗ trợ, chưa sẵn sàng train 12 lớp. Phiên bản segmentation v1 mới đã ghép đủ 43 ảnh Roboflow; bộ 130 ảnh ở liên kết này là snapshot chuẩn bị trước đó.
