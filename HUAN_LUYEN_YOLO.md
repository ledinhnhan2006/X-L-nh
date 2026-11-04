# Huấn luyện thêm lớp và chấm điểm theo bố trí

## Hai bài toán riêng

1. YOLO detection tìm loại đồ vật và khung bao. Huấn luyện thêm lớp cần ảnh và nhãn từng đồ vật.
2. Chấm điểm sai vị trí cần quy tắc nơi được phép đặt đồ vật và thông tin vùng sàn/bàn/giường/tủ. Chỉ có khung bao đồ vật chưa đủ để suy ra vị trí đúng.

Đã có chế độ YOLO theo vị trí và diện tích, công cụ vẽ vùng và bảng sửa quy tắc. Chưa huấn luyện vì chưa có dữ liệu. Xem training/CHUAN_BI_DU_LIEU.md để bắt đầu với 12 lớp đã chọn. Chế độ OpenCV cũ vẫn có thể chọn riêng.

## Chuẩn bị dữ liệu

Chọn danh sách lớp cuối cùng trước khi gán nhãn. Gán nhãn cả những lớp cũ muốn giữ; fine-tune chỉ với lớp mới không tự đảm bảo giữ lại các lớp COCO cũ. Thu thập ảnh đa dạng góc nhìn, ánh sáng, kích thước đồ vật, che khuất và cả phòng gọn lẫn bừa bộn. Có thể bắt đầu thử với vài trăm ví dụ mỗi lớp, nhưng số lượng này không đảm bảo chất lượng.

Dùng CVAT hoặc công cụ tương tự để vẽ khung bao, xuất định dạng Ultralytics YOLO detection. Tệp nhãn mỗi ảnh có các dòng `class_id center_x center_y width height`, tọa độ chuẩn hóa [0,1]. Kiểm tra trực quan nhãn trước khi train. Tách train/val/test theo phòng hoặc phiên chụp; không đưa các khung hình gần giống nhau sang nhiều tập.

Cấu trúc bộ dữ liệu:

```
dataset/
  data.yaml
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

`data.yaml` cần `path`, `train`, `val`, `test` và `names` theo đúng thứ tự lớp đã gán nhãn. Không tự thay đổi thứ tự sau khi tạo nhãn. Không có dữ liệu mẫu giả trong dự án này.

## Chạy train

Trong môi trường đã cài thư viện:

```bash
python train_yolo.py --data dataset/data.yaml --epochs 100 --batch 8 --device 0
```

`--device 0` chỉ dùng khi đã có GPU CUDA và PyTorch bản CUDA tương thích. Bản cài đặt dự án hiện dùng PyTorch CPU. Nếu chưa có GPU, dùng `--device cpu --batch 2`; việc huấn luyện có thể mất nhiều thời gian. Chạy thử ít epoch để kiểm tra đường dẫn/nhãn, sau đó mới train đầy đủ. Không dùng kết quả chạy thử để báo cáo chất lượng.

Script dùng mô hình nền yolo11n.pt, lưu best.pt và last.pt trong thư mục kết quả; không ghi đè trọng số nền. Validation được Ultralytics chạy trong quá trình train. Sau khi chọn mô hình bằng validation, đánh giá riêng tập test: precision/recall, mAP50-95 theo lớp, lỗi trên vật thể nhỏ và đồ vật bị che. Không chỉnh tham số theo kết quả test rồi tiếp tục báo cáo đó là test độc lập.

## Nạp mô hình vào web

Windows CMD, từ thư mục dự án và môi trường đã cài:

```bat
set YOLO_MODEL_PATH=C:\duong-dan\best.pt
.venv\Scripts\python.exe web_app.py --open-browser
```

Dừng máy chủ cũ trước khi chạy lại. Tên lớp trong giao diện lấy từ mô hình được nạp. Chỉ sử dụng tệp trọng số từ nguồn tin cậy.

## Nguyên tắc và giới hạn chấm điểm

Giai đoạn đầu: người dùng khoanh đa giác cho các vùng trên mỗi ảnh, hoặc dùng cấu hình cố định nếu camera không đổi. Mỗi lớp có vùng được phép, vùng bị cấm và mức phạt. Ví dụ chỉ là gợi ý, chưa phải quy tắc đã áp dụng: giày dưới sàn trong vùng kệ giày hợp lệ; giày trên giường không hợp lệ; sách trên bàn hoặc trong kệ hợp lệ.

Lưu trạng thái đúng/sai/chưa đủ thông tin cho từng phát hiện. Vật thể ngoài vùng được khai báo hoặc lớp chưa có quy tắc không được mặc nhiên coi là đúng vị trí. Không phát hiện được vật thể không đồng nghĩa phòng gọn hay 100 điểm.

Có thể đo tỷ lệ diện tích khung bao trên ảnh; đó là diện tích hình chiếu, không phải diện tích thật của đồ vật. Muốn chính xác hơn cần segmentation và hiệu chỉnh phối cảnh. Các vùng chồng lấp phải tính diện tích hợp để tránh đếm hai lần. Điểm neo đáy khung bao và mức chồng lấp vùng chỉ là xấp xỉ vị trí, đặc biệt dễ sai ở ảnh chụp xiên.

Mức phạt vị trí, diện tích đồ vật sai chỗ và chiếm lối đi cần hiệu chỉnh bằng ảnh được con người đánh giá. Mỗi khoản trừ nên hiển thị đồ vật nào, sai quy tắc nào và số điểm bị trừ. Chưa có dữ liệu/nhãn để huấn luyện hoặc hiệu chỉnh các mức phạt trong kho hiện tại.
