# Phân tích ảnh, rà soát vật thể và chấm điểm v5

Đây là cách tính tại trang chính của app. Các phép đo có thể lặp lại với cùng ảnh, đường bao và quy tắc. Điểm vẫn là một thang quy tắc thử nghiệm; chưa có bộ ảnh được nhiều người chấm độc lập để xác nhận nó phản ánh chính xác mức gọn của phòng.

## 1. Thao tác trong app

1. Chọn ảnh hoặc dùng ảnh mẫu, bật/tắt gợi ý YOLO, bấm **Phân tích ảnh**.
2. Xem **Ảnh từng bước** và **Thông số hình học** để biết ảnh đang được xử lý ra sao.
3. Khoanh **phạm vi đánh giá** nếu chỉ muốn chấm một phần phòng. Nếu không khoanh, toàn ảnh là phạm vi, gồm cả tường và đồ nội thất nhìn thấy.
4. Khoanh sàn, giường, bàn, tủ/kệ… bằng các đỉnh theo thứ tự quanh vùng. Chọn **Vùng này cần để trống** cho lối đi/vùng không được để đồ. Bấm lưu; có thể bỏ đỉnh cuối hoặc hủy nếu vẽ sai.
5. Kiểm tra từng gợi ý YOLO. Sửa tên, lớp, số lượng trong nhóm, vùng, đường bao và ghi chú. Dự đoán dưới 50% phải được người dùng xác nhận mới có thể bị tính điểm.
6. Với đồ bị bỏ sót: **Thêm vật bằng mắt thường**, vẽ quanh phần nhìn thấy, chọn lớp và xác nhận. Có thể dùng **Vật khác / chưa rõ** rồi tự xác nhận đúng/sai vị trí. Không biết thì để **Chờ xác nhận** hoặc **Bỏ qua**.
7. Đồ đúng chỗ nhưng bừa: đánh dấu **Đúng vị trí nhưng cần sắp xếp**. Số lượng tối đa trên bàn/kệ chỉ có hiệu lực khi bạn đặt giới hạn cụ thể.
8. Bấm **Chấm điểm ảnh**, xem từng khoản trừ rồi tải ZIP/JSON. Thay đổi thông tin sẽ làm điểm cũ hết hiệu lực; cần chấm lại trước khi tải báo cáo.

Nếu chỉ biết số lượng và vị trí, có thể hủy bước vẽ đường bao rồi nhập hai thông tin đó. App chỉ tính các tiêu chí đủ dữ liệu; không dùng diện tích khung vuông thay phần còn thiếu.

## 2. Đọc ảnh và các ảnh trung gian

App nhận JPG/PNG/WebP/BMP tối đa 14 MB, 25 triệu pixel. Pillow đọc ảnh, sửa hướng theo EXIF, chuyển RGB rồi `cv2.cvtColor(..., COLOR_RGB2BGR)`. Ảnh lớn được thu nhỏ giữ tỷ lệ, cạnh dài tối đa 1280 pixel. Pixel/px² trong bảng là trên **ảnh phân tích**; báo cáo ghi cả kích thước gốc và phân tích.

| Bước | Hàm | Công thức / mục đích |
|---|---|---|
| Ảnh xám | `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` | `Y = 0.299R + 0.587G + 0.114B` |
| Giảm nhiễu | `cv2.GaussianBlur(gray, (5,5), 0)` | Chập với `G(x,y) = exp(-(x²+y²)/(2σ²))/(2πσ²)`; OpenCV chọn σ từ kernel khi truyền 0 |
| Biên Canny | `cv2.Canny(blur, 80, 160)` | Gradient, triệt tiêu không cực đại, ngưỡng kép và nối cạnh; `|∇I| = √(Gx² + Gy²)` |
| Nhị phân/giãn biên | `cv2.threshold(edges,50,255,THRESH_BINARY)` và `cv2.dilate(..., ones((5,5)), iterations=2)` | Canny vốn nhị phân; giãn nối các nét gần nhau để tạo vùng bao |
| Tất cả contour | `cv2.findContours(binary, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` | Vùng cạnh ngoài, chưa phải danh sách đồ vật |
| Lọc contour | `cv2.contourArea(c) > 0.001 × W × H` | Bỏ vùng cạnh nhỏ hơn 0,1% ảnh; hiển thị trước và sau lọc |

**Edge density** `D = N_pixel_cạnh/(W×H)`. Ví dụ 8.000 pixel cạnh trong ảnh 400×500 thì `D=4%`. App cũng đo số contour/megapixel, diện tích trung bình và diện tích hợp vùng CV.

**Diện tích vùng CV** là vùng khép kín bởi các nét sau giãn; có thể bao nhiều đồ vật hoặc nền. Nó không phải diện tích vật thể và không thay mask đồ vật khi chấm.

## 3. Đường bao và hình học từng vật

YOLO chạy trên ảnh màu, song song với chuỗi chẩn đoán OpenCV. Ảnh xám/Canny không phải đầu vào thay thế cho YOLO.

- **YOLO segmentation:** `model.predict(...)` trả lớp, độ tin cậy, khung và instance mask. `scale_masks` loại padding letterbox, đưa mask về kích thước phân tích. Mask → `findContours(RETR_TREE, ...)` → contour ngoài/lỗ.
- **YOLO detection:** không có mask; `cv2.grabCut` ước lượng vùng trước/nền trong khung gợi ý, rồi tìm contour. Đây là ước lượng có thể sai. Nếu không đo được thì để trống, không lấy diện tích khung thay thế.
- **Thủ công:** các đỉnh bạn vẽ là đường bao. Máy chủ kiểm tra tọa độ `[0,1]`, đỉnh trùng, cạnh tự cắt và tính lại hình học. Không tin dùng giá trị diện tích do phía trình duyệt gửi lên.

| Đại lượng | Công thức / hàm |
|---|---|
| Diện tích | `A = Σ|contourArea(ngoài)| − Σ|contourArea(lỗ)|`; đa giác: `A = ½|Σ(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)|` |
| Chu vi | `P = Σ√((xᵢ₊₁−xᵢ)²+(yᵢ₊₁−yᵢ)²)`; `cv2.arcLength`, gồm biên lỗ |
| Tâm | `cx=M10/M00`, `cy=M01/M00`; `cv2.moments`, trừ moment lỗ |
| Độ tròn | `4πA/P²` |
| Độ đặc (solidity) | `A/A_bao_lồi`, với `cv2.convexHull` |
| Mức lấp đầy khung (extent) | `A/(rộng×cao)` |
| Tỷ lệ hình dạng | `rộng/cao` của khung trục ngang/dọc |
| Đường kính tương đương | `2√(A/π)` |
| Khung xoay | `cv2.minAreaRect`, `cv2.boxPoints`: tâm, kích thước, góc |
| Cạnh trong vật | `pixel Canny nằm trong mask/pixel mask` |

Số pixel foreground được xuất riêng với diện tích contour; hai số có thể hơi khác vì một bên đếm ô pixel, một bên đo đa giác. Khung trục/khung xoay chỉ giúp kiểm tra hình học, **không thay diện tích contour**.

Điểm đặt ước lượng lấy x trung vị của các pixel thuộc 5% thấp nhất mask và y thấp nhất mask. App đối chiếu điểm này với vùng bằng `cv2.pointPolygonTest`. Vùng nhỏ được ưu tiên khi bàn/giường nằm trong vùng sàn lớn. Hai vùng cùng diện tích có quy tắc xung đột tại điểm đặt thì báo chưa rõ; bạn có thể gán vùng hoặc xác nhận đúng/sai. Góc chụp có thể làm điểm đặt sai; kết luận của người rà soát được lưu riêng.

## 4. Vật nào được đưa vào điểm?

Mỗi dòng là một vật hoặc một nhóm; số lượng nhóm từ 1 đến 100. Độ tin cậy YOLO gốc luôn được giữ lại, kể cả khi sửa nhãn. Vật thêm thủ công không có độ tin cậy mô hình; app không tự gán 100%.

- **Gợi ý AI:** chỉ đánh giá tự động khi tin cậy ≥50%, có lớp và vị trí/quy tắc đủ rõ.
- **Đã xác nhận:** dùng thông tin người dùng rà soát; có thể xác nhận vị trí trực tiếp hoặc đối chiếu theo vùng.
- **Chờ xác nhận / chưa rõ:** không bị phạt.
- **Bỏ qua:** giữ trong báo cáo, loại khỏi mọi khoản phạt và diện tích chấm; không suy ra vật đó đã gọn.
- **Ngoài phạm vi:** không chấm. Vật có điểm đặt trong phạm vi hoặc ít nhất nửa mask nằm trong phạm vi được xem xét; diện tích chỉ lấy phần giao phạm vi.

`Vật cần sắp xếp` gồm vật sai vị trí hoặc vật được người dùng đánh dấu lộn xộn. Đồ đúng chỗ, không bị đánh dấu và không vượt giới hạn thì không bị trừ vì có nhiều hoa văn/cạnh.

Quy tắc mặc định sửa được trong app: chăn ở giường; gối ở giường/tủ; sách ở bàn/tủ; laptop ở bàn/tủ; ghế ở sàn; rác ở thùng rác; giày ở khu giày/tủ; túi/quần áo/túi xách/balo ở tủ; chai ở bàn/tủ. Muốn sách trên giường hoặc giày trên sàn được tính đúng thì chọn thêm ô tương ứng.

## 5. Công thức chấm điểm

Điểm nền 100, tổng mức trừ tối đa 60, điểm sàn 40:

```text
S = max(40, 100 − P_vị_trí − P_số_lượng − P_diện_tích − P_contour − P_biên)
```

**Không có vật đánh giá được:** `S = null`, giao diện hiện chưa đủ thông tin. Không dùng 40 hay 100 thay giá trị thiếu. Khi chỉ đánh giá được một phần, điểm chỉ có ý nghĩa trên phần đó.

### Vị trí — tối đa 12

```text
P_vị_trí = min(12, 2 × max(mức_nghiêm_trọng_của_vật_sai))
```

Mức lớp: rác 3; quần áo/giày/laptop 1,5; túi/chăn/túi xách/chai/balo/vật khác 1; gối/sách/ghế 0,5. Vật sai tại vùng cần trống có mức nhân 2. Lấy mức lớn nhất để phản ánh độ nghiêm trọng; số lượng tính riêng bên dưới. Vật đúng chỗ nhưng cần sắp xếp không bị phạt vị trí.

### Số lượng — tối đa 12

```text
P_số_lượng = min(12, N_vật_cần_sắp_xếp + N_vượt_giới_hạn_chưa_tính)
```

Số lượng nhóm được cộng ở đây, không nhân diện tích nhóm. Với từng giới hạn vùng/lớp: `vượt = max(0, số_lượng − giới_hạn)`. Trừ đi số vật cần sắp xếp đã tính trong nhóm đó. Nếu các giới hạn chồng nhau tại cùng vùng, lấy phần vượt bổ sung lớn nhất để tránh phạt trùng. Đây là cách bảo thủ, có thể không tính hết các nhóm vượt độc lập trong cùng vùng.

Vượt số lượng không tự chọn mask làm “vật thừa” để phạt diện tích. Với đồ bừa thật, người dùng cần đánh dấu cụ thể.

### Diện tích — tối đa 16

Tạo **hợp mask** vật cần sắp xếp, cắt theo phạm vi; phần chồng chỉ đo một lần. Lấy contour của hợp này, không cộng diện tích khung.

```text
r_sai = diện_tích_hợp_mask_cần_sắp_xếp / diện_tích_phạm_vi
r_cản = diện_tích_hợp_mask_cần_sắp_xếp_giao_vùng_cần_trống / diện_tích_vùng_cần_trống
P_diện_tích = min(16, 40 × max(0, r_sai − 0,10, r_cản − 0,02))
```

Không có vùng cần trống thì `r_cản=0`. Có thể chỉnh hai ngưỡng miễn 10%/2% trên giao diện. Lấy giá trị lớn hơn để tránh cộng trùng cùng diện tích vừa sai chỗ vừa cản lối.

### Contour / độ phân tán — tối đa 10

```text
k = số mảng rời của hợp mask cần sắp xếp có diện tích pixel ≥ max(4, 0,001 × A_phạm_vi)
P_contour = min(10, 2 × max(0, k − 1))
```

Dùng `cv2.connectedComponentsWithStats`. Nhiều mảng rời biểu thị đồ cần sắp xếp phân tán. Đây không phải số contour Canny toàn ảnh hay số vật thật; nhiều vật chạm nhau có thể thành một mảng.

### Đường biên — tối đa 10

```text
B = L_biên_hợp_mask_cần_sắp_xếp / (2√(πA_phạm_vi))
P_biên = min(10, 2 × max(0, B − 0,10))
```

`L` gồm biên ngoài/lỗ, bỏ các đoạn ngay mép ảnh/phạm vi do cắt vùng tạo ra. Không lấy nét hoa văn trong vật. Chỉ số chuẩn hóa theo phạm vi; hình dạng vật vẫn có thể làm biên dài hơn, nên khoản này được giới hạn và chỉ áp dụng với vật đã xác định cần sắp xếp.

Để hợp mask/đếm mảng/tính điểm ổn định theo độ phân giải, app đưa mask và vùng lên lưới cạnh dài 512, **giữ tỷ lệ ảnh**, rồi đo contour hình học. Đây là xấp xỉ raster có thể bỏ sót chi tiết rất nhỏ. Hình học từng vật trong bảng đo trên ảnh phân tích tối đa 1280; tỷ lệ hợp mask có thể lệch nhẹ so với cộng số liệu bảng.

Ví dụ giả sử các khoản trừ là `6 + 3 + 4 + 2 + 1,8`, điểm sẽ là `83,2/100`. Giao diện/JSON xuất số liệu đầu vào, công thức, mức trừ thô và mức sau giới hạn.

## 6. Vai trò của edge density và giới hạn phép chấm

Thảm, sách, chăn hoa và cửa chớp đều có thể nhiều cạnh dù đặt gọn. Canny không biết cạnh nào thuộc vật gì. Edge density, số contour CV và hình học toàn ảnh dùng để kiểm tra; chỉ contour/biên của **vật cần sắp xếp đã đủ thông tin** đi vào điểm. Nhận xét “đúng chỗ nhưng lộn xộn” cần người dùng xác nhận trong bản này.

Các tiêu chí diện tích, phân tán và biên có liên quan nhau. App giới hạn từng khoản để giảm tác động chồng; trọng số vẫn cần hiệu chỉnh bằng dữ liệu con người chấm. Đổi góc chụp, ánh sáng hoặc phạm vi có thể đổi điểm. App không đo diện tích sàn thật, vật bị che khuất, bụi hay vệ sinh chỉ từ các số liệu này.

Tỷ lệ “đã đánh giá” chỉ tính trên danh sách vật đã phát hiện/nhập. Kể cả 100% danh sách cũng không chứng minh đã thấy hết vật trong ảnh. Checkpoint segmentation hiện còn nhiều lỗi; hãy bổ sung, sửa hoặc bỏ qua những gợi ý không đáng tin.

## 7. Báo cáo và mã nguồn

ZIP xuất gồm `bao_cao.json`, `vat_the.csv`, `cac_buoc/` và hướng dẫn. JSON lưu SHA-256 ảnh đầu vào, kích thước, tham số xử lý, mô hình, mọi vật kể cả bỏ qua/chưa rõ, polygon, nguồn máy/người, quy tắc, phạm vi, ngưỡng và các khoản trừ. ZIP chứa ảnh đầu vào ở kích thước phân tích và các ảnh trung gian khi bạn chủ động tải báo cáo.

App giữ ảnh tạm trong RAM tối đa 1 giờ, tối đa 4 phiên ảnh. Không tự gửi ảnh tới Roboflow/dịch vụ AI ngoài; YOLO chạy trên máy chạy app. Tắt máy chủ/hết phiên thì cần phân tích lại ảnh.

| Tệp | Vai trò |
|---|---|
| `image_features.py` | Ảnh trung gian, edge density, contour CV và hình học |
| `object_contours.py` | Mask YOLO/GrabCut, contour ngoài/lỗ và diện tích |
| `scene_scoring.py` | Kiểm tra dữ liệu rà soát, hợp mask và 5 tiêu chí |
| `image_workbench.py` | Phiên ảnh trong RAM và xuất ZIP/CSV/JSON |
| `web_app.py` | YOLO và API `/api/workbench/analyze`, `/score`, `/export` |
| `web/index.html`, `web/app.js`, `web/styles.css` | Giao diện và tương tác rà soát |

Thông tin checkpoint/dataset: [MODEL_SEGMENTATION_V1.md](MODEL_SEGMENTATION_V1.md). Công thức cũ còn dùng ở `/legacy`: [PHUONG_PHAP_CHAM_DIEM.md](PHUONG_PHAP_CHAM_DIEM.md).
