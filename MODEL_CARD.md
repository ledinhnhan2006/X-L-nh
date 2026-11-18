# Checkpoint weights.pt đi kèm ứng dụng

- Nguồn: tệp weights.pt do chủ dự án cung cấp, huấn luyện trên Roboflow theo thông tin người dùng.
- Kích thước: 5.427.845 byte.
- SHA-256: `ac4c4b0ad994344b308a71728d275bcbbd2af8acdc082ecb04774613f3084c1a`.
- Ứng dụng dùng tệp này mặc định; YOLO_MODEL_PATH nếu được đặt sẽ ưu tiên hơn.
- Đã nạp và chạy suy luận trên CPU với ultralytics 8.4.202 và torch 2.15.0+cpu. Chưa kiểm tra trực tiếp môi trường Windows của người dùng.

## Các lớp thực tế, đúng thứ tự trong checkpoint

| ID | Tên lớp |
|---|---|
| 0 | bag |
| 1 | blanket |
| 2 | book |
| 3 | bottle |
| 4 | chair |
| 5 | clothes |
| 6 | footwear |
| 7 | handbag |
| 8 | laptop |
| 9 | pillow |
| 10 | trash |

**Không có backpack (balo).** Bảng quy tắc chấm điểm vẫn giữ đủ 12 lớp mục tiêu để có thể dùng checkpoint sau này; bảng đó không thay đổi số lớp mô hình nhận diện. Muốn thêm backpack cần có nhãn lớp này, kiểm tra lớp đã được đưa vào bản dữ liệu xuất và huấn luyện lại.

## Kiểm tra tích hợp

Nạp được đúng weights.pt, kiểm tra API trạng thái mô hình và suy luận thực tế trên các ảnh mẫu. Trên room.jpg tại ngưỡng 0,25, nhận được các dự đoán pillow, book và blanket. Đây là đầu ra dự đoán, chưa được xác minh là nhãn đúng cho mọi khung. App dùng phát hiện đó để áp dụng quy tắc vị trí và công thức v2 trong PHUONG_PHAP_CHAM_DIEM.md.

Không có bộ test độc lập được đính kèm checkpoint để tính lại precision, recall hay mAP. Số ảnh huấn luyện do người dùng báo là 43, chưa được xác minh từ bộ dữ liệu. Không tuyên bố mô hình hoặc điểm số đạt độ chính xác cụ thể. Giữ điều kiện giấy phép áp dụng cho checkpoint từ nguồn huấn luyện; tệp này không được tái cấp phép chỉ bằng giấy phép mã nguồn app.
