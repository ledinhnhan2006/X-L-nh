# 100 ảnh khởi đầu cho YOLO: 6 lớp đồ vật

Bộ này gồm đúng **100 ảnh khác nhau**, mỗi ảnh có ít nhất một khung bao thuộc các lớp dưới đây. Đã kèm nhãn YOLO detection; không phải nhãn do mô hình tự dự đoán. Tổng khoảng 16 MB.

| ID mới | Lớp | Ảnh train có lớp | Ảnh val có lớp | Tổng khung bao |
|---|---|---:|---:|---:|
| 0 | backpack — balo | 13 | 5 | 26 |
| 1 | handbag — túi xách | 18 | 6 | 42 |
| 2 | bottle — chai | 19 | 4 | 76 |
| 3 | chair — ghế | 35 | 5 | 136 |
| 4 | laptop — laptop | 14 | 3 | 23 |
| 5 | book — sách | 16 | 5 | 115 |

Một ảnh có thể chứa nhiều lớp, nên không cộng cột số ảnh để suy ra tổng. Có **80 ảnh train / 20 ảnh validation**, tổng **418 khung bao**. Tập validation nhỏ chỉ dùng thử quy trình, không phải phép đánh giá đáng tin cậy cho phòng thực tế. Chưa có tập test riêng.

## Cách lấy về máy

Tại https://github.com/ledinhnhan2006/X-L-nh chọn **Code → Download ZIP**, giải nén. Bộ ảnh nằm ở `training/starter100/`:

```
images/train/   80 ảnh
images/val/     20 ảnh
labels/train/  80 tệp nhãn
labels/val/     20 tệp nhãn
```

Mỗi dòng nhãn: `class_id center_x center_y width height`, tọa độ chuẩn hóa [0,1]. ID của bộ này chỉ có 0–5, KHÁC với ID trong bộ 12 lớp ở training/data.yaml. Không dùng nhầm tệp YAML.

## Chạy thử huấn luyện trên Windows

Sau khi đã chạy `start_windows.bat` để cài thư viện, mở CMD tại thư mục dự án:

```bat
.venv\Scripts\python.exe train_yolo.py --starter100 --epochs 3 --batch 2 --device cpu
```

`--starter100` tự tạo data-local.yaml với đường dẫn tuyệt đối theo máy của bạn. Không cần tự sửa đường dẫn. Ba epoch chỉ kiểm tra quy trình. Chưa có mô hình best.pt mới được huấn luyện sẵn trong kho. Khi muốn train lâu hơn, đổi epochs; CPU có thể rất chậm.

Huấn luyện này tạo mô hình 6 lớp, không tự thêm được các lớp chưa có nhãn. Ảnh có bối cảnh trong nhà và ngoài trời, dùng để thử nhận diện đồ vật; không có nhãn phòng gọn/bừa bộn hay đồ vật đúng/sai vị trí.

## Những gì bộ này chưa có

Chưa có nhãn cho **bag (túi đựng đồ), clothes (quần áo), blanket (chăn), pillow (gối), footwear (giày dép), trash (rác)**. Dù chúng có thể xuất hiện trong ảnh, không được coi là đã có nhãn. Không dùng bộ nhãn sáu lớp này để train trực tiếp cấu hình 12 lớp. Muốn mở rộng cần gán bổ sung tất cả đồ vật thuộc các lớp mới và chuyển lại ID cho đúng.

Các lớp đã có ở mô hình nền YOLO; bộ 100 ảnh này không chứng minh mô hình nhận tốt hơn hay mở rộng số lớp. Dữ liệu đến từ COCO val2017, tập vốn được dùng đánh giá mô hình COCO và có thể đã ảnh hưởng việc lựa chọn mô hình tiền huấn luyện. Không dùng kết quả trên 20 ảnh này để tuyên bố hiệu quả độc lập hoặc tính tổng quát cho căn phòng mới.

## Nguồn và bản quyền

- Nguồn ảnh gốc: [Microsoft COCO](https://cocodataset.org/#download), split val2017.
- Bản sao ảnh để tải qua GitHub: [zlzGithub-0801/COCO2017val](https://github.com/zlzGithub-0801/COCO2017val), ghim commit trong `stats.json` và từng URL trong `manifest.csv`.
- Nhãn YOLO chuyển đổi từ COCO: [Ultralytics coco2017labels.zip](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip). Tệp giấy phép đi kèm nhãn được giữ nguyên tại `ANNOTATION_SOURCE_LICENSE.txt`; mã SHA-256 của archive được lưu trong stats.json.
- Chỉ giữ sáu lớp: COCO ID 24,26,39,56,63,73 → ID mới 0,1,2,3,4,5.
- `manifest.csv` và `manifest.json` lưu ID ảnh COCO, URL nguồn gốc, URL bản sao, kích thước ảnh và SHA-256 từng ảnh. Không có ảnh do AI tạo hay nhãn giả.

Ảnh COCO có giấy phép ảnh Flickr tương ứng theo từng ảnh; giấy phép mã nguồn của dự án hoặc giấy phép bản chuyển nhãn không thay thế giấy phép ảnh. Dùng metadata ID ảnh COCO để tra tác giả/giấy phép gốc khi sử dụng ngoài thử nghiệm nghiên cứu. Không tuyên bố sở hữu các ảnh này.

## Đã kiểm tra

- 100 ảnh đều giải mã được; không trùng SHA-256.
- 100 tệp nhãn khớp tên ảnh; 418 khung bao, ID lớp và tọa độ hợp lệ.
- Train và validation không trùng tệp; cả hai tập đều có sáu lớp.
- Bộ nạp dữ liệu Ultralytics đọc đủ 80 ảnh/325 khung train và 20 ảnh/93 khung validation, không có ảnh/nhãn hỏng.
- Chưa kiểm tra trùng gần giống theo cảnh/phiên chụp; chưa rà từng khung bao bằng tay. Nhãn nguồn có thể có lỗi, cần rà lại trước huấn luyện chính thức.
