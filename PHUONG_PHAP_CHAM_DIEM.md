# Phương pháp chấm ảnh YOLO v3 — thang điểm nhẹ

## Mở app với mô hình đã huấn luyện của bạn

1. Tải bản mới từ GitHub: Code → Download ZIP. Giải nén vào thư mục mới, không làm mất thư mục/mô hình cũ.
2. Bản GitHub đã kèm `weights.pt` do bạn cung cấp. Không cần chép thêm để dùng mô hình này. Nếu có mô hình mới, thay tệp cạnh `web_app.py` rồi khởi động lại app.
3. Lần đầu chạy `start_windows.bat` để tạo môi trường và cài thư viện. Những lần sau nhấp đúp `start_app.bat`, không cần gõ CMD hoặc cài lại thư viện.
4. Nếu dùng môi trường cũ, có thể chạy trực tiếp `start_app.bat` khi `.venv` đã tồn tại và hoạt động trên máy Windows đó. Không sao chép `.venv` từ Linux/máy khác.
5. App mặc định tìm `weights.pt`; biến `YOLO_MODEL_PATH` nếu đã được đặt sẽ ưu tiên hơn. Muốn quay lại mặc định trong CMD, chạy `set YOLO_MODEL_PATH=` trước khi mở app. App hiển thị tên tệp và các lớp thực tế đã nạp.
6. Nếu không có tệp, mô hình không tương thích hoặc nạp lỗi, app báo lỗi và không tự dùng yolo11n.pt để thay thế. Gửi lỗi nguyên văn nếu xảy ra. Mô hình chỉ được xác minh trên máy bạn sau khi đặt checkpoint vào app.

Đã nạp và chạy được checkpoint Roboflow bạn cung cấp với phiên bản thư viện hiện có. Mô hình có 11 lớp và thiếu backpack; app hiển thị cảnh báo. Không tự thêm được một lớp bằng cách sửa danh sách tên.

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

## 2. Điểm nền và điều kiện được trừ

Điểm khởi đầu là **100**, điểm sàn là **40**. Đây là quy ước để giảm trừ quá mức, không phải độ chính xác YOLO hay thang đo đã được chứng minh khoa học.

Chỉ đánh giá vị trí khi độ tin cậy của phát hiện **từ 0,50 trở lên**. Dự đoán dưới 0,50 vẫn có thể xuất hiện nếu đạt ngưỡng hiển thị (mặc định 0,25), nhưng ghi là chưa đủ thông tin và không trừ điểm vị trí hoặc diện tích. Lớp lạ, ngoài vùng đã khoanh và vùng mơ hồ cũng không bị phạt.

Không phát hiện/không đánh giá được vật nào → **không trả điểm**, thay vì tự thưởng 100 hay điểm sàn 40.

## 3. Phạt vị trí nhẹ, có giới hạn theo loại

| Loại vật sai vị trí | Điểm phạt / vật |
|---|---:|
| Rác | 3 |
| Quần áo, giày dép, laptop | 1,5 |
| Túi đựng đồ, chăn, túi xách, chai, balo | 1 |
| Gối, sách, ghế | 0,5 |

Mỗi loại chỉ bị trừ tối đa **8 điểm**, bất kể có bao nhiêu phát hiện. Tổng phạt vị trí P tối đa **45 điểm**:

`P = min(45, tổng theo loại[min(8, tổng phạt các vật sai thuộc loại đó)])`.

Bảng từng vật hiển thị khoản trừ sau giới hạn loại, trước giới hạn tổng 45. Nếu loại đó đã đủ 8 điểm, vật tiếp theo cùng loại vẫn được đánh dấu sai nhưng không bị trừ vị trí thêm. Đây là cách hạn chế đếm nhiều vật nhỏ hoặc dự đoán lặp làm điểm giảm quá nhanh.

## 4. Diện tích chỉ trừ phần cần thiết

Tính diện tích hợp các khung bao của vật sai chỗ **đủ tin cậy**. Phần chồng lấp chỉ tính một lần, xấp xỉ bằng lưới 512×512.

`r = diện tích hợp các khung sai / diện tích ảnh`.

Miễn **10% diện tích đầu tiên** để giảm tác động của khung bao có nền. Chỉ trừ phần vượt 10%, tối đa 15 điểm:

`A = min(15, 30 × max(0, r − 0,10))`.

Ví dụ: r=5% → không trừ diện tích; r=25% → trừ 4,5 điểm; r=60% trở lên → tối đa 15 điểm. Không phải diện tích vật thật hoặc diện tích sàn, nên góc chụp vẫn ảnh hưởng.

## 5. Công thức tổng và diễn giải

`S = max(40, 100 − P − A)`, làm tròn 2 chữ số.

Ví dụ: một vật rác sai chỗ (3), một áo sai chỗ (1,5), khung sai chiếm 15% ảnh (phạt diện tích 1,5) → **94/100**. Đây là thang nhẹ theo lựa chọn thiết kế, không phải kết luận phòng sạch.

Nhiều vật sai chỉ thuộc một loại: phạt vị trí tối đa 8 + diện tích tối đa 15 → điểm không thấp hơn 77. Nhiều loại sai dày đặc: tổng phạt tối đa 45+15 → điểm sàn 40. Điểm sàn không có nghĩa mọi ảnh đều đạt mức gọn gàng.

Nhãn ở chế độ YOLO: từ 90 gọn gàng; từ 75 khá gọn; từ 55 hơi bừa bộn; dưới 55 bừa bộn. Chế độ OpenCV giữ thang cũ riêng.

Nếu có phát hiện chưa đủ thông tin, kết quả ghi **điểm tạm tính trên phần đã đánh giá**. Có lớp mục tiêu thiếu trong mô hình thì cảnh báo. Kể cả đủ tên lớp, mô hình có thể bỏ sót đồ vật; 100 chỉ nghĩa chưa có khoản trừ trong phần quan sát được, không xác nhận cả phòng gọn.

## Đã và chưa xác minh

Kiểm tra công thức, giới hạn điểm, mức phạt từng lớp, vùng chồng nhau, thiếu thông tin, thiếu checkpoint, API và thao tác giao diện. Đã kiểm tra với weights.pt thực tế bạn cung cấp: nạp mô hình, đọc 11 lớp, suy luận trên ảnh mẫu, chấm theo vùng và quy tắc, tải ảnh/báo cáo. Việc suy luận thành công không chứng minh độ chính xác trên ảnh mới; chưa có bộ ảnh gán nhãn độc lập để đánh giá precision/recall/mAP của checkpoint này. Chưa có đánh giá độc lập chứng minh điểm v3 chính xác hơn theo cảm nhận con người.
