# Chạy giao diện Góc Gọn

## Windows (máy của bạn)

1. Cài Python 3.12 từ python.org. Khi cài, bật Python Launcher và Add Python to PATH.
2. Giải nén toàn bộ dự án vào một thư mục (không chạy trực tiếp từ trong tệp ZIP).
3. Nhấp đúp `start_windows.bat`. Lần đầu cần mạng để tải thư viện, có thể mất vài phút.
4. Trình duyệt sẽ tự mở sau khi ứng dụng khởi động. Giữ cửa sổ lệnh đang mở.
5. Bấm **Thêm ảnh của bạn**, chọn ảnh JPG/PNG/WebP/BMP rồi bấm **Chấm điểm ảnh**.
6. Có thể bật **Nhận diện đồ vật bằng YOLO** để xem khung, tên và độ tin cậy.
7. Bấm **Tải ảnh kết quả** để lưu ảnh đường bao. Đóng cửa sổ lệnh để dừng.

Nếu trình duyệt không tự mở: trên CHÍNH máy đang chạy ứng dụng, nhập `http://127.0.0.1:8000` vào thanh địa chỉ. Địa chỉ này chỉ dùng trên máy đó; không phải đường dẫn môi trường đám mây. Nếu lỗi, giữ cửa sổ lệnh và chụp thông báo. Trình khởi chạy Windows chưa được thử trực tiếp trên Windows; backend và trình duyệt được kiểm tra trên Linux.

## Môi trường đám mây đã chuẩn bị

```bash
cd /workspace/X-L-nh
source /workspace/xlnh-venv/bin/activate
export MPLCONFIGDIR=/workspace/xlnh-setup/matplotlib
export YOLO_CONFIG_DIR=/workspace/xlnh-setup/yolo
python web_app.py
```

Máy chủ dùng cổng 8000, chỉ lắng nghe loopback theo mặc định. Giao diện onboarding hiện không cung cấp đường dẫn xem web này từ máy cá nhân. Để dùng trực tiếp, tải dự án và chạy trên máy của bạn. Không cần xuất bản môi trường thêm để làm điều này.

## Cách tính điểm

Giữ nguyên hàm `calculate_score` trong `chamdiem.py`:

- Ảnh xám → Gaussian 5×5 → Canny (80,160).
- Giãn cạnh bằng kernel 5×5, 2 lần; tìm đường bao ngoài.
- Giữ vùng có diện tích > 0,1% ảnh.

| Điểm phạt (đều tối đa 100) | Công thức | Trọng số |
|---|---|---|
| Cạnh toàn ảnh | min(mật độ cạnh × 300, 100) | 25% |
| Số vùng | min(số vùng × 3, 100) | 30% |
| Cạnh nửa dưới | min(mật độ cạnh nửa dưới × 300, 100) | 25% |
| Diện tích vùng | min(tỷ lệ diện tích × 200, 100) | 20% |

Điểm = 100 − tổng các điểm phạt nhân trọng số, giới hạn [0,100], làm tròn 2 chữ số. Từ 80: gọn gàng; từ 60: khá gọn; từ 40: hơi bừa bộn; dưới 40: bừa bộn. `room.jpg` đạt 65,30 điểm; ảnh JPG còn lại đạt 56,45.

Đây là thước đo thử nghiệm dựa vào đặc trưng ảnh, chưa được kiểm chứng trên bộ dữ liệu gọn/bừa bộn có nhãn. Hoa văn, góc chụp, ánh sáng và độ phân giải ảnh ảnh hưởng kết quả. Đường bao không tương đương đồ vật thực tế. Ảnh không phải căn phòng cũng nhận điểm nhưng điểm đó không có ý nghĩa đánh giá phòng.

## Vai trò YOLO

`testyolo.py` nạp trọng số `yolo11n.pt`, chạy suy luận trên `room.jpg`, in nhãn lớp, độ tin cậy, tâm và khung bao. Giao diện mới chạy cùng mô hình trên ảnh người dùng, dùng CPU và ngưỡng tin cậy 0,25; hiển thị khung nhận diện và bảng đồ vật. YOLO không tham gia công thức điểm, không được huấn luyện trong dự án để phân loại phòng gọn hay bừa bộn. Mô hình có thể bỏ sót hoặc nhận nhầm.

Ảnh tải lên chỉ được xử lý trong bộ nhớ; ứng dụng không chủ động lưu chúng hay gửi lên dịch vụ AI. Giới hạn tải lên 14 MB, 25 triệu pixel. Ảnh được xoay theo EXIF trước khi xử lý, ảnh màu/ảnh xám được chuyển về RGB.

## Kiểm tra

Khởi động `python web_app.py` ở một terminal. Trong terminal khác, cùng môi trường Python:

```bash
python -m unittest discover -s tests -v
```

Kiểm tra trang web, health, tải hai ảnh mẫu, điểm và tổng điểm phạt, YOLO, ảnh trắng không YOLO và dữ liệu tải lên không hợp lệ. Giao diện được kiểm tra thêm bằng Chromium: ảnh mẫu, tải tệp, bật/tắt YOLO và độ rộng màn hình điện thoại.
