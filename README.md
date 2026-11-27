# Góc Gọn — Phân tích ảnh và chấm điểm có rà soát

Thêm ảnh căn phòng, xem ảnh sau từng bước xử lý, sửa gợi ý YOLO và chấm điểm trên những vật đã đủ thông tin. App có thể dùng thủ công khi chưa có mô hình.

- Đặc trưng ảnh: edge density, số contour trước/sau lọc, diện tích vùng cạnh.
- Hình học vật thể: diện tích contour trừ lỗ, chu vi, tâm, điểm đặt, khung xoay, độ tròn, độ đặc, tỷ lệ rộng/cao.
- Rà soát: thêm vật, vẽ lại đường bao, sửa lớp, nhập số lượng, xác nhận vị trí, đánh dấu cần sắp xếp, ghi chú hoặc bỏ qua.
- Điểm v5: vị trí, số lượng, diện tích chiếm sai/cản lối, độ phân tán contour và đường biên. Có công thức và mức trừ tối đa.
- Tải ZIP gồm ảnh từng bước, bảng CSV vật thể và JSON ghi các khoản trừ, quy tắc, vùng cùng thông tin đã nhập.

## Mở trên Windows

1. Giải nén **toàn bộ** gói app vào một thư mục, ví dụ `D:\Goc-Gon`. Không chạy trong cửa sổ ZIP.
2. Nhấp đúp **`start_phan_tich_windows.bat`**. Gói tải đầy đủ có `weights_segmentation.pt`; launcher chọn mô hình này. Nếu thiếu checkpoint, phần phân tích ảnh và nhập thủ công vẫn dùng được.
3. Nếu chưa có Python 3.12, mở CMD chạy `py install 3.12`, rồi mở lại launcher. Lần đầu cần Internet để cài thư viện; giữ cửa sổ dòng lệnh mở trong lúc sử dụng.
4. Trình duyệt mở `http://127.0.0.1:8000`. Nếu chưa tự mở, nhập địa chỉ này trên **chính máy chạy app**.

Không cần chép `.venv` từ ZIP: launcher tạo môi trường trên Windows. Mô hình 12 lớp còn bỏ sót/nhận sai nhiều vật, vì vậy hãy rà soát dự đoán.

Nếu chỉ tải mã nguồn GitHub, đặt checkpoint cung cấp riêng dưới tên `weights_segmentation.pt` cạnh `web_app.py`. `start_segmentation_windows.bat` yêu cầu checkpoint này; `start_windows.bat`/`start_app.bat` dùng mô hình cũ khi không đặt `YOLO_MODEL_PATH`.

## Cách dùng

**Thêm ảnh → Phân tích ảnh → Khoanh vùng/rà soát vật → Chấm điểm → Tải báo cáo.**

Khoanh phạm vi muốn đánh giá; nếu không khoanh, mẫu số diện tích là toàn ảnh. Khoanh sàn, giường, bàn, tủ/kệ, khu giày, thùng rác hoặc lối đi. Chọn vật để sửa nhãn/số lượng hoặc bổ sung bằng mắt thường. Với vật không rõ, chọn **Chờ xác nhận** hoặc **Bỏ qua**. Có thể xác nhận đúng/sai vị trí thủ công khi chưa biết vùng phù hợp.

Vật đúng chỗ nhưng bừa: đánh dấu **Đúng vị trí nhưng cần sắp xếp**. Muốn giới hạn số đồ trên bàn/kệ: thêm giới hạn theo vùng. Hoa văn hoặc nhiều sách đặt đúng chỗ không tự làm giảm điểm.

Xem [hướng dẫn đầy đủ và công thức v5](PHUONG_PHAP_V5.md). Giao diện cũ tại `/legacy` dùng [công thức v4](PHUONG_PHAP_CHAM_DIEM.md). Điểm là thang quy tắc thử nghiệm, chưa hiệu chỉnh theo đánh giá của nhiều người. Diện tích là phép chiếu 2D, không phải m² sàn thật.

## Mô hình và dữ liệu

[Mô hình segmentation v1](MODEL_SEGMENTATION_V1.md) có 12 lớp: backpack, bag, blanket, book, bottle, chair, clothes, footwear, handbag, laptop, pillow, trash. Dataset Roboflow gồm 140 ảnh, phần lớn nhãn được hỗ trợ tự động; kết quả kiểm tra mask còn thấp. Đường bao có thể cần sửa bằng tay. Mô hình detection cũ có 11 lớp: [MODEL_CARD.md](MODEL_CARD.md).

Tài liệu bộ nháp trước khi ghép đủ 43 ảnh Roboflow: [training/segmentation_v1](training/segmentation_v1/README.md). Tập huấn luyện không cần có trên máy để chạy app.

## Chạy và kiểm tra trên máy phát triển

```bash
python -m pip install -r requirements-web.txt
YOLO_MODEL_PATH=weights_segmentation.pt python web_app.py --port 8765
TEST_BASE_URL=http://127.0.0.1:8765 TEST_MODEL_FILE=weights_segmentation.pt python -m unittest discover -s tests -v
```

Chọn cổng trống phù hợp môi trường. Kiểm tra HTTP cần app chạy tại `TEST_BASE_URL`. `image_features.py` trích xuất, `image_workbench.py` quản lý ảnh/báo cáo, `scene_scoring.py` tính điểm, `web_app.py` cung cấp API và `web/` là giao diện.
