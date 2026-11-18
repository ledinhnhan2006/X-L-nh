# Phương pháp chấm ảnh YOLO v2

## Mở app với mô hình đã huấn luyện của bạn

1. Tải bản mới từ GitHub: Code → Download ZIP. Giải nén vào thư mục mới, không làm mất thư mục/mô hình cũ.
2. Chép tệp `weights.pt` đã tải từ Roboflow vào **cùng thư mục với `web_app.py`**. Tệp của bạn không nằm sẵn trên GitHub; không cần tải nó lên GitHub.
3. Lần đầu chạy `start_windows.bat` để tạo môi trường và cài thư viện. Những lần sau nhấp đúp `start_app.bat`, không cần gõ CMD hoặc cài lại thư viện.
4. Nếu dùng môi trường cũ, có thể chạy trực tiếp `start_app.bat` khi `.venv` đã tồn tại và hoạt động trên máy Windows đó. Không sao chép `.venv` từ Linux/máy khác.
5. App mặc định tìm `weights.pt`; biến `YOLO_MODEL_PATH` nếu đã được đặt sẽ ưu tiên hơn. Muốn quay lại mặc định trong CMD, chạy `set YOLO_MODEL_PATH=` trước khi mở app. App hiển thị tên tệp và các lớp thực tế đã nạp.
6. Nếu không có tệp, mô hình không tương thích hoặc nạp lỗi, app báo lỗi và không tự dùng yolo11n.pt để thay thế. Gửi lỗi nguyên văn nếu xảy ra. Mô hình chỉ được xác minh trên máy bạn sau khi đặt checkpoint vào app.

App mới không kèm checkpoint Roboflow cá nhân. Phiên bản thư viện hiện giữ nguyên bản đã kiểm thử; khả năng nạp một checkpoint YOLO26 cụ thể cần kiểm tra với chính tệp đó.

## Cách chấm một ảnh

1. Chọn JPG/PNG/WebP/BMP hoặc kéo thả ảnh; tối đa 14 MB và 25 triệu pixel.
2. Giữ chế độ **YOLO — vị trí và diện tích** (mặc định).
3. Khoanh các vùng trên chính ảnh này: sàn, giường, bàn, tủ/kệ. Có thể thêm khu để giày, thùng rác, lối đi. Chọn loại vùng, bấm các đỉnh theo thứ tự quanh vùng, bấm Lưu vùng. Mỗi ảnh mới cần khoanh lại.
4. Mở bảng quy tắc để chọn nơi được đặt cho từng lớp. Mặc định giày ở khu giày/tủ, rác trong thùng rác; lối đi không được chọn cho lớp nào. Người dùng có thể sửa các nơi được phép.
5. Chọn ngưỡng tin cậy (mặc định 0,25), nhấn Chấm điểm ảnh. Hạ ngưỡng có thể tăng phát hiện lẫn nhận nhầm; không phải biện pháp nâng độ chính xác chắc chắn.
6. Xem kết quả và tải ảnh/báo cáo JSON. Khung xanh: đúng chỗ, đỏ: sai chỗ, vàng: chưa đánh giá được. Số # trên khung khớp số trong giải thích. Chấm đúng hay sai dựa vào dự đoán, không xác nhận đồ vật trong ảnh thực sự đúng nhãn.

## 1. Nhận diện và xác định vị trí

YOLO chạy trên ảnh, trả tên lớp, độ tin cậy và khung bao (x1,y1,x2,y2). Chỉ giữ dự đoán đạt ngưỡng. App không dùng mật độ cạnh OpenCV trong điểm YOLO; chế độ OpenCV cũ là lựa chọn riêng.

Vị trí xấp xỉ là điểm giữa đáy khung: `((x1+x2)/2, y2)`. Nếu nhiều vùng chứa điểm đó, ưu tiên vùng có diện tích nhỏ nhất; ví dụ bàn nhỏ được ưu tiên hơn đa giác sàn lớn. Hai vùng khác loại cùng diện tích chồng tại điểm neo → chưa đủ thông tin. Vật ngoài tất cả vùng hoặc lớp không khớp quy tắc → chưa đủ thông tin, không tự coi đúng hoặc sai. Đây là xấp xỉ 2D, có thể gán nhầm khi vật bị che, khung bao không sát hoặc góc chụp xiên.

Tập tên lớp chuẩn: bag, clothes, blanket, pillow, footwear, trash, handbag, laptop, book, bottle, chair, backpack. Chuẩn hóa chữ hoa/thường và nhận một số tên tương đương như shoes → footwear. Những tên khác cần đổi trong nhãn hoặc bổ sung ánh xạ; không tự đoán lớp.

## 2. Phạt theo loại vật sai chỗ: P

Mỗi vật đúng chỗ: 0 điểm phạt. Mỗi vật sai chỗ cộng mức sau:

| Loại vật | Điểm phạt vị trí / vật |
|---|---:|
| Rác | 10 |
| Quần áo, giày dép, laptop | 6 |
| Túi đựng đồ, chăn, túi xách, chai, balo | 4 |
| Gối, sách, ghế | 3 |

`P = min(60, tổng mức phạt các vật sai chỗ)`.

Các mức là quy ước thử nghiệm của app, không phải kết quả YOLO học được, không phải chuẩn vệ sinh. Bảng giải thích hiển thị mức phạt từng vật **trước khi áp dụng giới hạn tổng 60**. Rác được ưu tiên cao hơn sách do mục tiêu bố trí/vệ sinh của dự án. Chưa có bộ đánh giá con người để xác nhận các hệ số này.

## 3. Phạt diện tích sai chỗ: A

Tạo vùng hợp của tất cả khung bao đồ vật sai chỗ. Phần chồng lấp chỉ đếm một lần. Diện tích được xấp xỉ trên lưới chuẩn hóa 512×512 để giới hạn tài nguyên.

`r = diện tích hợp các khung sai / diện tích ảnh`.

`A = min(40, 200 × r)`.

Ví dụ các khung sai chiếm khoảng 5% ảnh thì bị trừ khoảng 10 điểm diện tích. Chiếm từ 20% ảnh trở lên bị trừ tối đa 40 điểm. Khung bao không phải mặt nạ đồ vật hay diện tích sàn thực; khoảng cách camera và góc chụp ảnh hưởng r.

## 4. Tổng điểm và trường hợp không đủ thông tin

`S = max(0, 100 − P − A)`, làm tròn 2 chữ số.

Ví dụ một vật rác sai chỗ (10), một áo sai chỗ (6), tổng khung sai chiếm khoảng 5% ảnh (10) → khoảng 74/100.

- Từ 80: gọn gàng; từ 60: khá gọn; từ 40: hơi bừa bộn; dưới 40: bừa bộn. Đây cũng là ngưỡng quy ước thử nghiệm.
- Không có vật nào đánh giá được → không trả điểm (hiển thị dấu —).
- Có vật chưa rõ lớp/vị trí → ghi **điểm tạm tính trên phần đã đánh giá**, hiển thị số đánh giá được và số chưa đủ thông tin; không coi phần chưa biết là đúng chỗ.
- Có lớp mục tiêu thiếu trong mô hình → cảnh báo. Dù đủ tên lớp, mô hình vẫn có thể bỏ sót vật thể; không nhìn thấy một vật không đồng nghĩa vật đó không có trong ảnh.
- 100 chỉ nghĩa không có vi phạm trong các phát hiện đã đánh giá được. Không chứng minh cả phòng gọn.
- Độ tin cậy YOLO dùng để lọc phát hiện, không nhân thành điểm gọn gàng và không được hiểu là xác suất đúng đã hiệu chuẩn.

## Đã và chưa xác minh

Kiểm tra công thức, giới hạn điểm, mức phạt từng lớp, vùng chồng nhau, thiếu thông tin, thiếu checkpoint, API và thao tác giao diện. Kiểm thử suy luận dùng yolo11n.pt qua cấu hình tường minh trong môi trường phát triển, không đại diện cho checkpoint Roboflow của bạn. Chưa nhận được tệp weights.pt cá nhân để kiểm tra khả năng nạp và độ chính xác của mô hình đó. Chưa có đánh giá độc lập chứng minh điểm v2 chính xác hơn theo cảm nhận con người.
