# Góc Gọn — Chấm điểm ảnh phòng với YOLO

Đã kèm `weights.pt` do bạn cung cấp; app tự dùng mô hình này. Lần đầu chạy `start_windows.bat`; lần sau mở `start_app.bat`. App hỗ trợ thêm ảnh, khoanh vùng, quy tắc vị trí, mức phạt theo lớp và diện tích, tải ảnh kết quả và báo cáo JSON.

**Hướng dẫn sử dụng và công thức v4 — diện tích contour, điểm nền 100, sàn 40:** [PHUONG_PHAP_CHAM_DIEM.md](PHUONG_PHAP_CHAM_DIEM.md).

Mô hình hiện có 11 lớp, **thiếu backpack (balo)**. Xem [MODEL_CARD.md](MODEL_CARD.md) để biết nguồn và kiểm tra. Đây là điểm thử nghiệm; không có điểm nếu không đánh giá được vật thể.

Diện tích ở chế độ YOLO được ước lượng bằng GrabCut → contour → cv2.contourArea, không lấy diện tích box. App hiển thị đường bao và trường hợp chưa đo được. Xem PHUONG_PHAP_CHAM_DIEM.md để biết giới hạn.

**Chuẩn bị dữ liệu instance segmentation:** [bộ nháp 130 ảnh (100 cũ + 30 mới)](training/segmentation_v1/README.md), có polygon nguồn và mặt nạ SAM hỗ trợ, chưa sẵn sàng train 12 lớp. Chưa ghép 43 ảnh Roboflow do thiếu bản xuất dataset.
