# Bắt đầu từ chưa có ảnh: 12 lớp đồ vật trong phòng

## 1. Chốt cách gọi và phân biệt các lớp

| ID | Tên nhãn | Tiếng Việt và phạm vi |
|---|---|---|
| 0 | bag | Túi đựng đồ/túi mua sắm, không phải túi xách cá nhân hay balo |
| 1 | clothes | Quần áo rời, gấp hoặc treo; không gồm chăn, gối |
| 2 | blanket | Chăn/mền |
| 3 | pillow | Gối |
| 4 | footwear | Giày và dép; mỗi chiếc là một khung riêng |
| 5 | trash | Rác rời nhìn thấy rõ: giấy vo, vỏ đồ ăn… Không gán mọi chai/túi là rác |
| 6 | handbag | Túi xách cá nhân; không gồm balo/túi mua sắm |
| 7 | laptop | Laptop; chưa gồm màn hình hay thùng máy tính để bàn |
| 8 | book | Sách |
| 9 | bottle | Chai, kể cả chai rỗng. Đặt sai chỗ do quy tắc vị trí quyết định |
| 10 | chair | Ghế |
| 11 | backpack | Balo |

Một đồ vật không gán đồng thời nhiều lớp. Nếu cần desktop, vỏ chai bỏ đi hay đôi giày làm một đối tượng, cần đổi quy ước trước khi gán nhãn. Các quy ước trên là mặc định ban đầu, có thể sửa theo mục tiêu dự án.

## 2. Thu thập ảnh

Bắt đầu với bộ thử nghiệm nhỏ có khoảng 50–100 ví dụ được gán nhãn mỗi lớp để kiểm tra quy trình. Sau đó tăng lên vài trăm ví dụ/lớp, ưu tiên các kiểu đang nhận sai; không có con số nào đảm bảo chất lượng. Một ảnh có thể chứa nhiều lớp.

Tự chụp nhiều phòng hoặc nhiều phiên chụp khác nhau. Có ảnh ban ngày/ban đêm, gần/xa, vật nhỏ/lớn, nhiều màu và kiểu dáng, vật bị che một phần. Thu thập cả phòng gọn, bừa bộn và ảnh nền không có lớp mục tiêu. Đừng chỉ chụp các đồ vật ở vị trí sai: YOLO cần học nhận diện chúng ở cả vị trí đúng.

Tách ảnh theo phòng hoặc phiên chụp thành train/validation/test, hướng tới tỷ lệ khoảng 70/20/10. Không chia ngẫu nhiên các ảnh gần như giống nhau hoặc khung hình sát nhau trong video sang nhiều tập. Giữ nguyên test để đánh giá cuối cùng. Nếu chỉ có một phòng, kết quả không chứng minh mô hình dùng tốt cho phòng mới.

## 3. Gán nhãn trong CVAT

1. Mở https://app.cvat.ai trên trình duyệt, tạo tài khoản nếu chưa có.
2. Tạo project, thêm đủ 12 nhãn đúng tên và thứ tự ở bảng trên.
3. Tạo task trong project và tải ảnh lên; có thể tạo task riêng cho các nhóm train/val/test.
4. Mở job để gán nhãn. Dùng Rectangle vẽ khung sát từng đồ vật, chọn đúng nhãn. Khung không bao luôn các đồ vật bên cạnh. Với vật bị che, dùng quy ước nhất quán bao phần quan sát được; không đoán vật không nhìn thấy.
5. Gán TẤT CẢ đồ vật thuộc 12 lớp nhìn thấy trong mỗi ảnh. Bỏ sót nhãn khiến mô hình học sai. Đánh dấu ảnh đã kiểm tra; ảnh âm tính thực sự không có lớp mục tiêu có tệp nhãn rỗng theo định dạng xuất.
6. Nhờ người khác kiểm tra hoặc tự xem lại một lượt, đặc biệt các cặp bag/handbag/backpack và blanket/clothes.
7. Xuất dữ liệu dạng **Ultralytics YOLO Detection** hoặc YOLO tương thích: mỗi dòng `class_id center_x center_y width height`, các tọa độ nằm trong [0,1]. Tên định dạng trong menu CVAT có thể thay đổi. Nếu CVAT xuất thứ tự ID khác, cập nhật `names` trong YAML để khớp đúng tệp nhãn; không chỉ đổi tên lớp rồi giữ ID sai.
8. Sắp xếp ảnh và nhãn cùng tên vào `dataset/images/{train,val,test}` và `dataset/labels/{train,val,test}`. Ví dụ `images/train/001.jpg` đi với `labels/train/001.txt`.

Đây là nhãn khung bao cho YOLO, khác với vùng sàn/giường/bàn/tủ bạn khoanh trong giao diện chấm điểm. Không cần gán 4 loại vùng này làm lớp đồ vật trong bộ dữ liệu hiện tại.

## 4. Cấu hình và train

Sao chép `training/data.yaml`, sửa `path` thành đường dẫn tuyệt đối tới thư mục dataset của bạn. Ví dụ Windows: `path: C:/Users/ADMIN/Documents/room_dataset`. Kiểm tra thứ tự `names` đúng với ID trong dữ liệu xuất.

Trong thư mục dự án, sau khi cài thư viện bằng trình khởi động:

```bat
.venv\Scripts\python.exe train_yolo.py --data training/data.yaml --device cpu --epochs 3 --batch 2
```

Lần 3 epoch chỉ kiểm tra dữ liệu và cài đặt, không phải mô hình hoàn chỉnh. Sau khi kiểm tra được, chạy huấn luyện chính (ví dụ 100 epoch, điều chỉnh theo validation). CPU có thể rất chậm. Muốn dùng GPU: chuyển bộ dữ liệu và mã sang máy/Colab có GPU, cài PyTorch CUDA phù hợp, dùng `--device 0`. Bản cài Windows hiện ưu tiên PyTorch CPU, không tự có khả năng CUDA.

Huấn luyện xong, kiểm tra precision/recall và mAP theo từng lớp trên validation. Đánh giá best.pt trên tập test riêng và xem ảnh dự đoán. Đặc biệt kiểm tra rác, đồ nhỏ, đồ bị che. Model 12 lớp này không tự giữ các lớp cũ như bed/vase/potted plant; muốn giữ lớp nào phải đưa lớp và ảnh nhãn tương ứng vào bộ dữ liệu cuối cùng.

## 5. Dùng mô hình mới trong giao diện

Mở CMD tại thư mục dự án, dừng web cũ rồi chạy:

```bat
set YOLO_MODEL_PATH=C:\duong-dan-thuc-te\best.pt
.venv\Scripts\python.exe web_app.py --open-browser
```

Dùng đường dẫn best.pt thực tế được in ra sau khi train. Không cần ghi đè yolo11n.pt. Giao diện tự lấy tên lớp từ mô hình mới.

## 6. Khoanh vùng và chấm điểm

- Chọn ảnh. Trong **Cách chấm điểm**, chọn **YOLO — vị trí và diện tích**.
- Phần khoanh vùng xuất hiện phía trên. Chọn sàn/giường/bàn/tủ, bấm các đỉnh theo thứ tự quanh vùng rồi bấm **Lưu vùng**. Có thể vẽ nhiều vùng cùng loại.
- Mở **Quy tắc vị trí cho 12 loại đồ vật**, bật các nơi được đặt cho từng lớp. Mặc định chỉ là đề xuất; ví dụ giày được để ở sàn hoặc tủ, còn rác không được để ở các vùng đó. Chưa có loại vùng thùng rác riêng.
- Nhấn **Chấm điểm ảnh**. Xem từng đồ vật, vùng được gán, đúng/sai/chưa đủ thông tin, diện tích khung bao và tổng điểm trừ.
- Ảnh mới sẽ xóa vùng cũ. Quy tắc sửa trong trang được giữ cho tới khi tải lại trang; vùng/quy tắc chưa được lưu thành cấu hình lâu dài.

Dùng điểm giữa đáy khung bao để ước lượng vật đang ở trên vùng nào; nếu nhiều vùng chứa điểm đó thì ưu tiên vùng nhỏ hơn. Đây không phải suy luận 3D và có thể gán nhầm ở ảnh chụp xiên.

Điểm thử nghiệm = 100 − min(60, số vật sai × 6) − min(40, diện tích hợp các khung sai / diện tích ảnh × 200). Diện tích hợp được xấp xỉ trên lưới 512×512 để tránh đếm đôi phần chồng lấp. Đồ vật chưa có quy tắc hoặc nằm ngoài vùng đã khoanh không được tính là đúng. Không có đồ vật đánh giá được thì không trả điểm. Chỉ chấm trên các đồ vật phát hiện được, nên 100 không chứng minh toàn bộ phòng gọn. Các mức phạt cần hiệu chỉnh với đánh giá thực tế từ con người; chưa phải thang đo đã được kiểm chứng.
