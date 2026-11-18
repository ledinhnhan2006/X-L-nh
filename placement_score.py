"""Chấm điểm vị trí từ phát hiện YOLO và vùng do người dùng khoanh."""
import math
import cv2
import numpy as np

CLASSES = {
    'bag': ('Túi (túi đựng đồ)', ['cabinet']),
    'clothes': ('Quần áo', ['cabinet']),
    'blanket': ('Chăn', ['bed']),
    'pillow': ('Gối', ['bed', 'cabinet']),
    'footwear': ('Giày dép', ['shoe_area', 'cabinet']),
    'trash': ('Rác', ['bin']),
    'handbag': ('Túi xách', ['cabinet']),
    'laptop': ('Máy tính (laptop)', ['desk', 'cabinet']),
    'book': ('Sách', ['desk', 'cabinet']),
    'bottle': ('Chai', ['desk', 'cabinet']),
    'chair': ('Ghế', ['floor']),
    'backpack': ('Balo', ['cabinet']),
}
ZONES = {'floor': 'Sàn', 'bed': 'Giường', 'desk': 'Bàn', 'cabinet': 'Tủ / kệ', 'shoe_area': 'Khu để giày', 'bin': 'Thùng rác', 'walkway': 'Lối đi'}
ALIASES = {'clothing': 'clothes', 'shoe': 'footwear', 'shoes': 'footwear', 'slipper': 'footwear',
           'slippers': 'footwear', 'garbage': 'trash', 'rubbish': 'trash', 'computer': 'laptop'}


# Thang điểm nhẹ: giới hạn tác động của số vật và khung bao lớn.
BASE_SCORE = 100
MIN_SCORE = 40
CLASS_CAP = 8
POSITION_CAP = 45
AREA_CAP = 15
AREA_ALLOWANCE = .10
ASSESSMENT_CONFIDENCE = .50
CLASS_PENALTIES = {'bag':1, 'clothes':1.5, 'blanket':1, 'pillow':.5, 'footwear':1.5, 'trash':3,
                   'handbag':1, 'laptop':1.5, 'book':.5, 'bottle':1, 'chair':.5, 'backpack':1}

def canonical_class(name):
    key = str(name).strip().casefold().replace(' ', '_').replace('-', '_')
    return ALIASES.get(key, key)

def validate_settings(zones, rules):
    if not isinstance(zones, list) or len(zones) > 40:
        raise ValueError('Danh sách vùng không hợp lệ (tối đa 40 vùng).')
    checked = []
    for zone in zones:
        if not isinstance(zone, dict) or zone.get('kind') not in ZONES:
            raise ValueError('Loại vùng không hợp lệ.')
        points = zone.get('points')
        if not isinstance(points, list) or not 3 <= len(points) <= 100:
            raise ValueError('Mỗi vùng cần 3–100 đỉnh.')
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                raise ValueError('Tọa độ vùng phải nằm trong [0,1].')
        poly = np.array(points, dtype=np.float32)
        # Đa giác đơn: loại bỏ cạnh cắt nhau, kể cả cạnh chồng hoặc chạm nhau.
        def cross(a,b,c):
            u,v = b-a,c-a
            return float(u[0]*v[1]-u[1]*v[0])
        def intersects(a,b,c,d):
            ab1,ab2,cd1,cd2 = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
            if ab1*ab2 < 0 and cd1*cd2 < 0:
                return True
            for val,p,u,v in [(ab1,c,a,b),(ab2,d,a,b),(cd1,a,c,d),(cd2,b,c,d)]:
                if abs(val) < 1e-9 and np.all(p >= np.minimum(u,v)-1e-9) and np.all(p <= np.maximum(u,v)+1e-9):
                    return True
            return False
        for i in range(len(poly)):
            for j in range(i+1,len(poly)):
                if j == i+1 or (i == 0 and j == len(poly)-1):
                    continue
                if intersects(poly[i],poly[(i+1)%len(poly)],poly[j],poly[(j+1)%len(poly)]):
                    raise ValueError('Vùng có cạnh tự cắt. Hãy vẽ lại theo thứ tự quanh vùng.')
        if abs(cv2.contourArea(poly)) < .0001 or len({tuple(p) for p in points}) != len(points):
            raise ValueError('Vùng quá nhỏ hoặc có đỉnh trùng nhau.')
        checked.append((zone['kind'], poly))
    if not isinstance(rules, dict) or set(rules) != set(CLASSES):
        raise ValueError('Cần quy tắc cho đủ 12 lớp đồ vật.')
    for allowed in rules.values():
        if not isinstance(allowed, list) or any(z not in ZONES for z in allowed):
            raise ValueError('Quy tắc vị trí không hợp lệ.')
    return checked


def score_placement(detections, image_shape, zones, rules):
    polygons = validate_settings(zones, rules)
    h,w = image_shape[:2]
    # Raster chuẩn hóa giữ chi phí tính diện tích có giới hạn; diện tích chỉ xấp xỉ.
    mask = np.zeros((512,512), np.uint8)
    rows, wrong, evaluated, raw_penalty = [], 0, 0, 0
    class_penalties = {}
    for det in detections:
        key = canonical_class(det['name'])
        x1,y1,x2,y2 = [float(v) for v in det['box']]
        x1,x2 = np.clip([x1/w,x2/w],0,1)
        y1,y2 = np.clip([y1/h,y2/h],0,1)
        anchor = (float((x1+x2)/2), float(y2))
        matches = [(abs(cv2.contourArea(p)),kind) for kind,p in polygons if cv2.pointPolygonTest(p,anchor,False)>=0]
        # Vùng nhỏ hơn được ưu tiên khi sàn lớn bao quanh bàn/giường.
        matches.sort()
        region = matches[0][1] if matches else None
        ambiguous = len(matches)>1 and abs(matches[0][0]-matches[1][0])<1e-6 and matches[0][1]!=matches[1][1]
        state, reason = 'unknown', 'Chưa có quy tắc cho lớp này.'
        row_penalty = 0
        if key in CLASSES:
            if det.get('confidence', 0) < ASSESSMENT_CONFIDENCE:
                reason = 'Độ tin cậy dưới 50%; cần kiểm tra, không trừ điểm.'
            elif ambiguous:
                reason = 'Hai vùng cùng diện tích chồng lấp; cần khoanh lại.'
            elif region is None:
                reason = 'Điểm giữa đáy khung bao nằm ngoài các vùng đã khoanh.'
            else:
                evaluated += 1
                state = 'correct' if region in rules[key] else 'wrong'
                reason = ('Đúng vùng được phép: ' if state=='correct' else 'Không được đặt tại: ') + ZONES[region]
                if state == 'wrong':
                    wrong += 1
                    raw_penalty += CLASS_PENALTIES[key]
                    row_penalty = min(CLASS_PENALTIES[key], max(0, CLASS_CAP-class_penalties.get(key,0)))
                    class_penalties[key] = class_penalties.get(key,0) + row_penalty
                    left,top = np.floor(np.array([x1,y1])*512).astype(int)
                    right,bottom = np.ceil(np.array([x2,y2])*512).astype(int)
                    mask[top:bottom,left:right] = 1
        rows.append({**det, 'class_key':key, 'display_name':CLASSES.get(key,(det['name'],))[0],
                     'region':region, 'state':state, 'reason':reason,
                     'position_penalty':row_penalty,
                     'base_position_penalty':CLASS_PENALTIES.get(key, 0) if state == 'wrong' else 0,
                     'area_ratio':float(max(0,x2-x1)*max(0,y2-y1))})
    area = float(mask.mean())
    deductions = [min(POSITION_CAP,sum(class_penalties.values())), min(AREA_CAP,max(0,area-AREA_ALLOWANCE)*30)]
    unknown = len(rows)-evaluated
    return dict(score=round(max(MIN_SCORE,BASE_SCORE-sum(deductions)),2) if evaluated else None,
                detections=rows, evaluated=evaluated, unknown=unknown, wrong=wrong,
                wrong_area_ratio=area, deductions=deductions, raw_position_penalty=raw_penalty,
                method='placement_v3_gentle', baseline=BASE_SCORE, minimum_score=MIN_SCORE,
                class_penalties=class_penalties, class_cap=CLASS_CAP,
                area_allowance=AREA_ALLOWANCE, assessment_confidence=ASSESSMENT_CONFIDENCE, coverage=round(evaluated/len(rows),3) if rows else 0,
                provisional=True,
                note='Thang điểm nhẹ: nền 100, sàn 40; tối đa 8 điểm mỗi loại, miễn 10% diện tích sai chỗ. Chỉ đánh giá phát hiện tin cậy từ 50%. Điểm thử nghiệm trên đồ vật nhận diện và đánh giá được; chưa xác nhận độ chính xác. ' +
                     ('Có đồ vật chưa đủ thông tin vị trí hoặc quy tắc. ' if unknown else '')+
                     'Không phát hiện đồ vật không chứng minh phòng gọn. Diện tích là diện tích khung bao trên ảnh.')
