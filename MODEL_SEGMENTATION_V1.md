# YOLO11 Nano Instance Segmentation — phiên bản 1

Dataset và lượt train thử đầu tiên đã hoàn tất trên Roboflow. Checkpoint được chọn cho app là bản Ultralytics fine-tune từ COCO chạy trên máy đám mây sau khi bản hosted đầu tiên không đạt kiểm tra. App đọc mặt nạ của mô hình để đo contour/diện tích; công thức trừ điểm và điểm giữa đáy box dùng xác định vị trí giữ như tài liệu chấm điểm.

- Dataset: [cham-diem-phong-segmentation/1](https://app.roboflow.com/le-dinh-nhan/cham-diem-phong-segmentation/1), 140 ảnh: 127 train, 6 validation, 7 test.
- Đã ghép đủ 43 ảnh gốc: 33 ảnh trùng bộ rooms100, 10 ảnh bổ sung. Có 30 ảnh bổ sung từ lần chuẩn bị trước, nên tổng không cộng trùng thành 173.
- 1.407 annotation đối tượng ở bản COCO chuẩn bị; Roboflow xuất 1.430 dòng polygon do một số mặt nạ có phần rời. Hai ảnh âm tính đã được kiểm tra trực quan.
- Mô hình chọn: YOLO11 Nano Segmentation, khởi tạo từ `yolo11n-seg.pt` COCO chính thức, fine-tune 20 epoch ở 512 pixel; app suy luận ở 640 pixel. Augmentation chỉ áp dụng train.
- Training ID: `local-coco-finetune-v1`. Model ID: `local-coco-finetune-v1`.
- Checkpoint: `weights_segmentation.pt`, 6,022,052 byte; SHA-256 `272d5a1fcf46f698d92d157302c4d890a8b73b2ad0cfbd90e91221c67441d9d1`.
- Lượt Roboflow ban đầu cấu hình 100 epoch có mask mAP50 test 0,0202, nhận diện rất yếu trên ảnh mẫu nên không dùng làm mô hình chính. Bản thay thế được khởi tạo COCO qua Ultralytics; không khẳng định nguyên nhân chính xác của kết quả hosted chỉ từ checkpoint.
- Đã nạp bằng Ultralytics, kiểm tra task `segment`, đủ 12 tên lớp, chạy validation trên 7 ảnh test và chấm ảnh qua nhánh mask của app trên Linux CPU. File batch Windows được cung cấp; chưa thực thi trực tiếp trên Windows.

## Mở ứng dụng

Giải nén toàn bộ gói app vào một thư mục. Đóng cửa sổ app cũ, rồi nhấp đúp `start_segmentation_windows.bat`. Nếu chưa có `.venv` trong thư mục này, script gọi trình cài có sẵn; cần Python 3.12. Khi trang mở, trạng thái phải hiện `weights_segmentation.pt (segmentation)` và 12 lớp. Chọn ảnh, khoanh vùng, sửa quy tắc nếu cần rồi chấm ảnh.

Gói app tải về đã kèm `weights_segmentation.pt`. Nếu chỉ lấy mã nguồn từ GitHub, đặt checkpoint segmentation đã huấn luyện vào cạnh `web_app.py` dưới tên này; launcher chọn nó bằng YOLO_MODEL_PATH. `start_windows.bat`/`start_app.bat` vẫn dùng lựa chọn mô hình mặc định cũ.

Checkpoint COCO fine-tune được chọn hiện được giữ trong gói tải về; project Roboflow vẫn giữ lượt hosted T1 để đối chiếu. Việc tải checkpoint mới trở lại Roboflow đã bị kiểm tra phê duyệt tự động chặn vì chưa có ủy quyền rõ cho bước tải mô hình mới lên dịch vụ. Dataset trên Roboflow đã đủ 140 ảnh và lượt train hosted đã hoàn tất.

## Ánh xạ lớp trong checkpoint

| ID | Lớp |
|---|---|
| 0 | backpack |
| 1 | bag |
| 2 | blanket |
| 3 | book |
| 4 | bottle |
| 5 | chair |
| 6 | clothes |
| 7 | footwear |
| 8 | handbag |
| 9 | laptop |
| 10 | pillow |
| 11 | trash |

Giữ nguyên ánh xạ trong data.yaml/checkpoint; app đọc model.names để lấy tên lớp. Không đổi ID bằng cách sửa danh sách tên.

## Đánh giá trên tập test của phiên bản này

| Chỉ số Ultralytics | Giá trị |
|---|---:|
| metrics/precision(B) | 0.2277 |
| metrics/recall(B) | 0.2612 |
| metrics/mAP50(B) | 0.1856 |
| metrics/mAP50-95(B) | 0.1448 |
| metrics/precision(M) | 0.2249 |
| metrics/recall(M) | 0.2302 |
| metrics/mAP50(M) | 0.1709 |
| metrics/mAP50-95(M) | 0.1145 |
| fitness | 0.2594 |

Các số này đo mức khớp với nhãn nháp trong 7 ảnh giữ lại, không xác nhận độ chính xác trên phòng mới. Balo chưa có mẫu trong validation/test; giày dép và túi xách chưa có trong test. Nhãn dùng COCO, box do người dùng gán và SAM hỗ trợ; mới xem tổng quan, chưa kiểm tra đầy đủ mọi biên và đồ vật. Mặt nạ có thể thiếu, gộp hoặc tách vật. Cần ảnh mới và nhãn kiểm tra độc lập để đánh giá khả năng áp dụng thực tế.

## Diện tích trong app

`prediction.masks.data` → bỏ đệm letterbox/scale_masks → mặt nạ ở kích thước ảnh gốc → ngưỡng 0,5 → findContours(RETR_TREE) → cộng contourArea đường bao ngoài, trừ lỗ. Phần chồng của đồ sai vị trí chỉ tính một lần khi phạt diện tích. Trường báo cáo: `yolo_segmentation_contour`. Thiếu mask hoặc tin cậy dưới 0,50 thì không dùng diện tích box thay thế.

Xem [công thức chấm điểm](PHUONG_PHAP_CHAM_DIEM.md). Mặt nạ 2D chưa cung cấp diện tích m² hoặc tự xác nhận đúng vị trí/gọn gàng.
