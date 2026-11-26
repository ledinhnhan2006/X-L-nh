# Dự án segmentation đã tạo trên Roboflow

**Cập nhật phiên bản 1:** đã ghép đủ 43 ảnh gốc, đưa 140 ảnh vào Dataset và hoàn tất train YOLO11 Nano Segmentation; bản chọn cho app được fine-tune lại từ COCO bằng Ultralytics. Xem [checkpoint và báo cáo mới](../../MODEL_SEGMENTATION_V1.md). Số liệu/chặn mạng bên dưới là snapshot của lần chuẩn bị 130 ảnh trước đó.

Dự án mới: https://app.roboflow.com/le-dinh-nhan/cham-diem-phong-segmentation

**Đã tải 130 ảnh và nhãn polygon nháp lên Roboflow. Chưa huấn luyện checkpoint mới.** Dự án detection gốc và `weights.pt` của ứng dụng được giữ nguyên.

## Đã thực hiện

- Tạo project **Instance Segmentation** riêng trong workspace `le-dinh-nhan`.
- Lấy lại đủ **969 bounding box của 43 ảnh** đã gán trong project `cham-diem-phong` phiên bản 1, qua API được xác thực. Không lấy nhãn từ checkpoint.
- Đối chiếu được ảnh gốc của **33/43 ảnh** với bộ `rooms100` trên máy. Đúng tên tệp và kích thước; không dùng ảnh chụp màn hình làm dữ liệu thay thế.
- Chuyển nhãn của 33 ảnh này thành **642 mặt nạ nháp**: 110 nhãn khớp lớp/khung với polygon COCO nguồn ở IoU > 0,50; 532 nhãn dùng box do người dùng gán làm prompt cho SAM2.1 Tiny. Loại 4 mặt nạ không đạt kiểm tra hình học.
- Ghép với phần ảnh còn lại của bộ 100 ảnh phòng cũ + 30 ảnh mới. 33 ảnh nói trên đã nằm trong bộ 100, nên không cộng trùng thành 163 ảnh.
- Tải **130 ảnh duy nhất**, gồm **1.086 đối tượng có polygon**, vào batch `segmentation-draft-v1`. Có 6 ảnh chưa có gợi ý nhãn; giữ chúng chờ gán nhãn, không coi là ảnh nền âm tính.
- Đọc lại API: Roboflow xác nhận 130 ảnh; kiểm tra 4 ảnh đầu đều có `type: polygon` và danh sách điểm, không chỉ có khung bao.
- Kiểm tra ZIP, giải mã toàn bộ ảnh, đối chiếu kích thước/SHA-256, ID lớp, tọa độ hữu hạn nằm trong ảnh và diện tích polygon dương.

Nhãn được nhập với `addToDataset=false`. Mở **Annotate** để xem batch và sửa polygon. Bộ này chưa được xác nhận hoàn chỉnh hoặc đưa vào Dataset để Generate phiên bản huấn luyện. Trường `split` lưu ở Source không chứng minh đã có một phiên bản dataset hoặc tập kiểm tra hoàn chỉnh.

## Phần còn thiếu

**10 ảnh gốc còn lại chưa tải được** vì chính sách mạng hiện tại chặn `source.roboflow.com` và `storage.googleapis.com`. Nhãn box của chúng đã được lấy và lưu trong `human_boxes_43.json`; không thể tạo đường bao đúng khi thiếu ảnh gốc. Danh sách tệp chờ nằm trong `info.pending_original_images` của COCO JSON.

Đã lưu bản nháp cấu hình thêm hai miền trên, giữ nguyên quyền API và các miền hiện có. Bản nháp chưa tự áp dụng vào môi trường đang chạy; người sở hữu cần lưu/xuất bản qua giao diện. Không cần gửi lại API key trong chat.

**Chưa bắt đầu train**: còn ảnh gốc cần ghép, polygon cần rà soát và thiếu dữ liệu cân bằng giữa các lớp. Hiện 655 nhãn là sách, nhưng chỉ có 3 balo và 6 rác; không thể coi việc khai báo 12 tên lớp là đủ dữ liệu cho 12 lớp. Khi lấy đủ 10 ảnh, cần chuyển thêm các nhãn có sẵn, rà soát nhãn thiếu/trùng và bổ sung ví dụ cho lớp ít mẫu trước khi chốt tập train/validation/test theo cảnh.

Việc xem các trang tổng quan của 33 ảnh chưa thay thế việc kiểm tra chi tiết từng vật thể. SAM tách đường bao theo prompt, không tự xác nhận lớp và không phát hiện mọi vật chưa được gán. Nhãn tự động có thể sai, thiếu hoặc gộp nhiều vật; polygon ngoài có thể lấp lỗ. Không dùng kết quả đánh giá trên chính nhãn tự tạo để khẳng định độ chính xác độc lập trên phòng mới.

## Số nhãn hiện có

| Lớp | Số đối tượng |
|---|---:|
| bag | 23 |
| clothes | 32 |
| blanket | 50 |
| pillow | 90 |
| footwear | 7 |
| trash | 6 |
| handbag | 10 |
| laptop | 23 |
| book | 655 |
| bottle | 20 |
| chair | 167 |
| backpack | 3 |

Thứ tự category ID trong COCO JSON bắt đầu từ 1 theo bảng trên. Khi xuất YOLO từ Roboflow phải giữ ánh xạ `names`/ID do phiên bản đó xuất; không tự lấy ID COCO gốc làm ID YOLO.

## Các tệp

- `roboflow_segmentation_DRAFT_130.zip`: ảnh gốc và `_annotations.coco.json` để nhập lại vào Roboflow khi cần; **bộ nháp**, không phải trọng số đã train.
- `_annotations.coco.json`: polygon và nguồn từng nhãn; mọi đối tượng vẫn có `needs_review: true`.
- `human_boxes_43.json`: 969 nhãn box gốc, kể cả 10 ảnh đang chờ tải.
- `manifest.json`: hash/kích thước ảnh, nguồn, ID ảnh đã lưu trên Roboflow và trạng thái cần kiểm tra.
- `stats.json`: số liệu, phần còn thiếu và trạng thái chưa train.
- `rejected_masks.json`: 4 mặt nạ không đạt kiểm tra, kèm prompt gốc để sửa.
- `previews/`: trang xem nhanh đường bao của 33 ảnh đã khôi phục nhãn; không đưa preview vào train.

Nguồn ảnh COCO và giấy phép polygon như [bộ nháp trước](../segmentation_v1/README.md). Annotation gốc của người dùng lấy từ project Roboflow công khai có giấy phép CC BY 4.0. Không đưa API key, URL tải có chữ ký hoặc phản hồi API chứa thông tin đăng nhập vào GitHub.

Ứng dụng hiện vẫn dùng detection + GrabCut. Sau khi có checkpoint segmentation đã kiểm tra, cần đọc đầu ra `masks` trong ứng dụng để chấm diện tích bằng mặt nạ của mô hình. Chỉ đổi tệp `.pt` chưa hoàn tất bước tích hợp này.
