# Bộ nháp instance segmentation: 100 ảnh phòng cũ + 30 ảnh mới

**Chưa phải bộ dữ liệu hoàn chỉnh để train 12 lớp. Chưa huấn luyện checkpoint mới.**

Đã chuẩn bị 130 ảnh thật, 793 nhãn đối tượng gồm 802 polygon sau khi loại 11 gợi ý sai rõ. Bộ này giúp tiết kiệm công gán nhãn, không thay thế việc kiểm tra đủ mọi đồ vật trong ảnh. Có 7 ảnh chưa có nhãn gợi ý nào; không được tự coi chúng là ảnh nền âm tính đã xác minh.

## Tải và mở trong Roboflow

1. Tải `roboflow_segmentation_DRAFT_130.zip` ở thư mục này bằng Download raw file trên GitHub.
2. Tạo project **Instance Segmentation** mới trên Roboflow; giữ project detection cũ.
3. Upload ZIP. Bên trong có 130 ảnh và `_annotations.coco.json` theo định dạng COCO Instance Segmentation. Nếu giao diện không nhận ZIP, giải nén rồi tải cả thư mục images và JSON cùng nhau.
4. Mở Annotate, kiểm tra polygon, sửa lớp/biên và bổ sung đồ vật còn thiếu. Không coi nhãn có sẵn là ảnh đã được kiểm tra hoàn tất.
5. Bổ sung ảnh rác và các lớp ít ví dụ. Chia theo phòng/phiên chụp thành train/validation/test sau khi hoàn tất nhãn; chưa chia tự động để tránh rò rỉ cảnh.
6. Generate phiên bản và xuất **YOLO Instance Segmentation**, giữ nguyên thứ tự names/ID do Roboflow xuất.

Ảnh đầy đủ nằm trong ZIP; thư mục images giữ ở máy chuẩn bị và không nhân đôi trong Git. Các trang previews giúp xem nhanh đường viền, không đưa preview vào tập train.

## Nguồn ảnh và phạm vi dữ liệu cũ

- 100 ảnh cũ là bộ `training/rooms100` đã chuẩn bị trong GitHub, KHÔNG phải bản xuất 43 ảnh bạn đã gán trên Roboflow.
- 30 ảnh mới là ảnh phòng khách chưa có trong cả rooms100 lẫn starter100, kiểm tra bằng ID và SHA-256. Cùng nguồn COCO val2017 qua bản sao GitHub ghim commit trong manifest.
- 43 ảnh tại `https://app.roboflow.com/le-dinh-nhan/cham-diem-phong/1` chưa lấy được: kết nối từ môi trường này bị proxy chặn 403, chưa kiểm tra được quyền đăng nhập của dịch vụ. Không có API key/đăng nhập Roboflow được cấu hình.
- Cần ZIP dataset của phiên bản đó để ghép đúng ảnh và nhãn đã gán. `weights.pt` không cung cấp lại toàn bộ ảnh/nhãn gốc.

## Các nhãn được tạo thế nào?

**631 nhãn nguồn:** polygon COCO đã chuyển sang YOLO segmentation bởi Ultralytics cho 6 lớp: book, bottle, chair, laptop, handbag, backpack. Được chuyển sang COCO JSON để nhập Roboflow. Nhãn nguồn vẫn cần kiểm tra theo phạm vi 12 lớp của dự án; không đảm bảo đã gán mọi vật xuất hiện.

**162 gợi ý tự động còn lại:** dùng `weights.pt` hiện có để gợi ý lớp/khung ở ngưỡng 0,45, rồi **SAM2.1 Tiny** tạo mask trong vùng đó. Chỉ bổ sung nhóm lớp chưa có trong nhãn COCO nguồn để hạn chế trùng. SAM không xác nhận tên lớp; nếu YOLO gọi nhầm mèo là quần áo, SAM có thể khoanh đúng con mèo nhưng nhãn vẫn sai.

Đã xem các trang preview, kiểm tra riêng các trường hợp nghi ngờ và loại 11 nhãn sai rõ (mèo/quần áo, laptop/giày, áo đang mặc/chăn, chân người/gối). Có kiểm tra mặt nạ tự động trùng cùng lớp ở IoU > 0,90; lần này không phát hiện thêm cặp phải bỏ. Chưa rà từng đồ vật trên toàn bộ 130 ảnh ở mức gán nhãn đầy đủ, nên số ảnh được xác nhận hoàn chỉnh vẫn là 0.

Màu preview: **xanh lam = nhãn COCO nguồn**, **cam = gợi ý tự động**. Mọi annotation có `needs_review: true`. Một đối tượng có thể gồm nhiều polygon rời; gợi ý có lỗ bên trong được đánh dấu `holes_need_review`, vì xuất polygon ngoài có thể lấp lỗ.

## Số nhãn hiện tại

| Lớp | Nhãn nguồn COCO | Gợi ý tự động sau lọc |
|---|---:|---:|
| bag | 0 | 13 |
| clothes | 0 | 28 |
| blanket | 0 | 46 |
| pillow | 0 | 70 |
| footwear | 0 | 5 |
| trash | 0 | 0 |
| handbag | 10 | 0 |
| laptop | 22 | 0 |
| book | 392 | 0 |
| bottle | 28 | 0 |
| chair | 174 | 0 |
| backpack | 5 | 0 |

**Thiếu trash hoàn toàn; footwear/backpack/handbag rất ít.** Không có gợi ý không có nghĩa ảnh không có vật đó. Nếu train ngay trên nhãn thiếu, mô hình có thể học coi vật chưa gán là nền. Không biến bộ nháp này thành bộ train chỉ bằng cách đổi tên JSON hoặc thêm names vào YAML.

## Tệp kiểm kê

- `annotations.draft.coco.json`: annotation hỗ trợ gán nhãn, chưa đầy đủ.
- `manifest.json`: nguồn, ID COCO, kích thước và SHA-256 từng ảnh.
- `review_manifest.csv/json`: số nhãn và trạng thái cần kiểm tra theo ảnh.
- `stats.json`: thống kê lớp, trạng thái chưa sẵn sàng train và hash các artifact nguồn.
- `manual_rejections.json`, `rejected_annotations.json`: quyết định loại nhãn, có dữ liệu để xem lại/khôi phục.
- `previews/`: 7 trang xem nhanh tất cả 130 ảnh sau lọc.

## Huấn luyện sau khi có dữ liệu đã rà soát

Script mới `train_segmentation.py` dùng mô hình nền `yolo11n-seg.pt`, không dùng checkpoint detection cũ để giả thành segmentation. Kiểm tra dữ liệu xuất từ Roboflow trước:

```bat
.venv\Scripts\python.exe train_segmentation.py --data dataset_seg/data.yaml --validate-only
```

Data.yaml cần path và train/val/test chỉ đúng tới thư mục images tương ứng. Giữ nguyên names/ID xuất từ Roboflow. Script kiểm tra đủ 12 lớp trong train, polygon chuẩn hóa, diện tích dương và nhãn khớp ảnh; không tự xác nhận chất lượng nhãn.

Sau khi kiểm tra thành công, thử quy trình 3 epoch:

```bat
.venv\Scripts\python.exe train_segmentation.py --data dataset_seg/data.yaml --epochs 3 --batch 2 --device cpu
```

Ba epoch không phải kết quả huấn luyện chính thức. Bản cài Windows hiện dùng PyTorch CPU; muốn GPU cần PyTorch CUDA tương thích hoặc môi trường GPU khác. Kết quả lưu trong runs/segment, không ghi đè weights.pt hiện tại.

App chấm điểm hiện dùng detection + GrabCut. Khi có best.pt segmentation đã kiểm tra, cần tích hợp đầu ra masks vào app để dùng trực tiếp; chỉ đổi checkpoint mà không sửa bước đọc mask vẫn chưa hoàn tất chuyển đổi.

## Nguồn artifact

- Ảnh: COCO val2017, bản sao GitHub `zlzGithub-0801/COCO2017val` tại commit trong URL của manifest. Giấy phép từng ảnh theo metadata COCO/Flickr, không bị thay thế bởi giấy phép mã nguồn app.
- Polygon nguồn: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip ; giữ giấy phép kèm archive tại SOURCE_ANNOTATION_LICENSE.txt.
- Mặt nạ gợi ý: SAM2.1 Tiny tại https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt ; hash trong stats.json. Không đưa mô hình SAM dung lượng lớn vào bản tải app.
- Mô hình gợi ý lớp: weights.pt do người dùng cung cấp; giữ nguyên checkpoint app.

Dữ liệu COCO có thể đã ảnh hưởng việc tiền huấn luyện/chọn mô hình nền. Không dùng kết quả validation trên ảnh này để khẳng định độ chính xác độc lập cho phòng mới. Cần ảnh phòng mới có nhãn kiểm tra riêng.
