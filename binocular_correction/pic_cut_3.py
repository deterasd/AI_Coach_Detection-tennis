import cv2
import os
# 影片路徑
video_path = r"C:/Users/chen/Pickleball_Project/trajectory/chen__trajectory/trajectory__63/chen__63_45.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't open the video")
    exit()

# 讀取影片基本資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片資訊: {width}x{height}, fps={fps}, 總幀數={total_frames}")

# 輸出資料夾
output_dir = r"C:/Users/chen/yolo/output_image"
os.makedirs(output_dir, exist_ok=True)

img_counter = 0

# 建立可調整大小的視窗，避免畫面太大顯示不完整
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("影片播放完畢")
        break

    # 顯示當前幀
    cv2.imshow('Video', frame)

    # 根據 fps 設定 waitKey (1000/fps 毫秒)，讓播放速度接近原影片
    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == 27:  # Esc 退出
        print("Escape hit, closing...")
        break
    elif key == ord('s'):  # s 截圖
        img_name = os.path.join(output_dir, f"opencv_frame_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} 保存成功")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
