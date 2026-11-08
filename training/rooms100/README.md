# 100 ảnh phòng thật để gán nhãn trong Roboflow

**Ảnh thật, chưa gán nhãn 12 lớp. Không phải bộ dữ liệu đã hoàn chỉnh để train.**

Có 100 ảnh khác nhau: 65 ảnh phòng ngủ/góc giường và 35 ảnh phòng khách/góc sinh hoạt. Đây là bộ mới, khác với `training/starter100` gồm ảnh vật thể COCO ở nhiều bối cảnh. Tổng ảnh khoảng 14 MB. Các ảnh có cảnh gọn và một số cảnh nhiều đồ, nhưng chưa cân bằng gọn/bừa bộn hoặc số ví dụ mỗi lớp.

## Tải và đưa vào Roboflow

1. Tải tệp **roboflow_upload_100.zip** trong thư mục này bằng nút Download raw file trên GitHub, hoặc tải toàn dự án qua Code → Download ZIP.
2. Giải nén ZIP riêng; bên trong có đúng 100 ảnh. Nếu dùng toàn dự án, chọn đúng thư mục `training/rooms100/images`, KHÔNG chọn bộ starter100 hoặc ảnh preview.
3. Vào app.roboflow.com, tạo project **Object Detection**.
4. Chọn Upload Data, tải 100 ảnh lên. Đợi tải xong rồi lưu.
5. Mở Annotate, tạo 12 tên lớp trong `classes.txt` và vẽ khung từng đồ vật. Chỉ gán lớp khi nhìn rõ và phân biệt được.
6. Mỗi ảnh cần gán tất cả đồ vật thuộc các lớp mục tiêu, không chỉ gán một đồ vật. Ảnh không có lớp nào chỉ được dùng như ảnh âm tính sau khi đã kiểm tra kỹ.

Không có tệp nhãn rỗng giả hoặc dự đoán YOLO được đóng vai nhãn người gán trong bộ này. Không chạy lệnh train trước khi gán nhãn và xuất dữ liệu từ Roboflow.

## Đủ 12 lớp chưa?

**Chưa xác nhận đủ 12 lớp và chưa đủ ví dụ để huấn luyện đáng tin cậy.**

- Các lớp sách, chai, ghế, laptop, balo, túi xách được đối chiếu với annotation COCO nguồn để ưu tiên tuyển ảnh.
- Chăn, gối, quần áo và giày có ví dụ quan sát được trong ảnh xem trước; chưa có khung bao cho các lớp này. Quần áo đang mặc và quần áo rời cần phân biệt theo quy ước dự án: ưu tiên quần áo rời.
- **Túi đựng đồ (bag)** và **rác (trash)** chưa được xác nhận rõ, cần bổ sung ảnh mục tiêu. Không tự gọi chai rỗng, giấy hoặc đồ vứt trên giường là rác khi chưa xác định được.
- Sáu lớp nguồn cũng có thể rất ít ảnh. Có một lớp xuất hiện không nghĩa là đã đủ dữ liệu để học lớp đó.

Xem `coverage.csv` để biết bằng chứng sơ bộ và những lớp cần bổ sung. Danh sách này là kiểm kê ảnh, KHÔNG phải nhãn huấn luyện. Nên thêm ảnh phòng bừa bộn có túi mua sắm, rác rõ ràng, giày dép và quần áo rời trong nhiều góc nhìn trước khi train 12 lớp. Không dùng 100 ảnh này để khẳng định chất lượng mô hình.

## Xem trước và chia tập

Có bốn trang preview_*.jpg, mỗi trang gồm 25 ảnh và số room_XXX để đối chiếu. Đã xem các trang ảnh và loại ảnh ngoài trời/cận cảnh thú cưng/ảnh ghép không phù hợp. Vẫn có ảnh góc giường hoặc người/thú nuôi trong phòng; kiểm tra kỹ trước khi gán nhãn.

Chưa chia train/val/test vì cần gán nhãn và gom các ảnh cùng phòng/phiên chụp trước. Không tự chia ngẫu nhiên các góc gần giống của cùng phòng sang train và test. Việc không trùng tệp không đảm bảo không trùng cảnh.

## Nguồn

- Ảnh gốc: [Microsoft COCO val2017](https://cocodataset.org/#download).
- Bản sao: [zlzGithub-0801/COCO2017val](https://github.com/zlzGithub-0801/COCO2017val), commit `075a8fd1f1ec671e4e469ee8b5ab6badcbda3db1`.
- `manifest.csv` / `manifest.json`: ID COCO, URL ảnh gốc, URL tải, kích thước và SHA-256 từng ảnh.
- Ảnh được tuyển từ các mẫu có annotation giường/sofa; sau đó xem ảnh để loại cảnh không phù hợp. Sáu lớp kiểm kê trong coverage dựa trên bản chuyển nhãn của Ultralytics coco2017labels.zip.
- Ảnh COCO giữ giấy phép Flickr tương ứng theo từng ảnh; giấy phép mã nguồn dự án không áp dụng thay cho giấy phép ảnh. Tra ID COCO để xem tác giả và giấy phép gốc khi cần phân phối/sử dụng theo điều kiện ảnh.

**Đã kiểm tra:** 100 ảnh giải mã được, 100 SHA-256 khác nhau, ZIP chứa đúng 100 ảnh và manifest khớp tệp. Chưa gán nhãn hoặc huấn luyện mô hình trên bộ này.
