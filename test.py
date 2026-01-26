from ultralytics import YOLO
import cv2

# 載入你訓練好的 Pose 模型
model = YOLO("C:/Users/chen/yolo/runs/pose/train2/weights/best-2.pt")

# 測試一張圖片
img_path = "C:/Users/chen/yolo/runs/pose/train2/weights/test4.jpg"
results = model(img_path, conf=0.1, show=False)  # 降低 conf

for result in results:
    print("=== YOLO 輸出結果 ===")
    print("Bounding boxes:", result.boxes.xyxy.cpu().numpy())
    print("Keypoints:", result.keypoints.xy)

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        keypoints = result.keypoints.xy[0].cpu().numpy()
        print("Detected keypoints array shape:", keypoints.shape)

        # 取前四個點
        if keypoints.shape[0] >= 4:
            pts = keypoints[:4]
            print("Paddle 4 points:", pts)

            # 讀取影像
            frame = cv2.imread(img_path)
            pts_int = pts.astype(int)

            # 畫出多邊形（球拍拍面）
            cv2.polylines(frame, [pts_int], isClosed=True, color=(0,255,0), thickness=2)

            # 畫出四個 keypoints（小紅點）
            for (x, y) in pts_int:
                cv2.circle(frame, (x, y), 5, (0,0,255), -1)

            # 顯示結果
            cv2.imshow("Paddle Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("⚠️ 偵測到的點數不足 4 個，模型輸出 shape:", keypoints.shape)
    else:
        print("⚠️ 沒有偵測到任何 keypoints！")


"""
from ultralytics import YOLO
import cv2

# 載入你訓練好的 Pose 模型
model = YOLO("C:/Users/chen/yolo/runs/pose/train2/weights/best-2.pt")

# 測試一張圖片
img_path = "C:/Users/chen/yolo/runs/pose/train2/weights/test2.jpg"
results = model(img_path, show=False)  # show=False，避免YOLO內建視窗干擾

for result in results:
    keypoints = result.keypoints.xy  # 每個 keypoint 的 (x, y) 座標
    bboxes = result.boxes.xyxy       # 邊界框 (xmin, ymin, xmax, ymax)

    print("bbox:", bboxes)
    print("keypoints:", keypoints)

    # 取第一個物件的四個點
    if len(keypoints) > 0:
        pts = keypoints[0].cpu().numpy()[:4]  # 你的球拍拍面四個角點
        print("Paddle 4 points:", pts)

        # 讀取影像
        frame = cv2.imread(img_path)
        pts_int = pts.astype(int)

        # 畫出多邊形（球拍拍面）
        cv2.polylines(frame, [pts_int], isClosed=True, color=(0,255,0), thickness=2)

        # 畫出四個 keypoints（小紅點）
        for (x, y) in pts_int:
            cv2.circle(frame, (x, y), 5, (0,0,255), -1)

        # 顯示結果
        cv2.imshow("Paddle Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""