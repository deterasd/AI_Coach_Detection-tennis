
import cv2
import os
import sys
from pathlib import Path

def convert_to_h264(input_path):
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return False
        
    print(f"Processing: {input_path}")
    
    # Read Input
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("Failed to open video")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0: fps = 30
    
    # Temp output
    temp_out = input_path.with_name(input_path.stem + "_fixed_h264.mp4")
    
    # H.264 Codec
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(temp_out), fourcc, fps, (width, height))
        if not out.isOpened():
             print("avc1 failed, trying H264")
             fourcc = cv2.VideoWriter_fourcc(*'H264')
             out = cv2.VideoWriter(str(temp_out), fourcc, fps, (width, height))
             
        if not out.isOpened():
            print("Failed to init H264 VideoWriter")
            return False
            
    except Exception as e:
        print(f"Error initializing writer: {e}")
        return False
        
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        count += 1
        if count % 50 == 0:
            print(f"Converted {count}/{total} frames...", end='\r')
            
    cap.release()
    out.release()
    print(f"\nDone! Saved to: {temp_out.name}")
    
    # Backup original and replace
    backup = input_path.with_name(input_path.name + ".bak")
    try:
        if backup.exists(): os.remove(backup)
        os.rename(input_path, backup)
        os.rename(temp_out, input_path)
        print("Replaced original file (backup saved as .bak)")
        return True
    except Exception as e:
        print(f"Error renaming files: {e}")
        return False

if __name__ == "__main__":
    files = [
        "knn_dataset_new/player6/player6_1/outdoor6__1_side_segment_processed.mp4",
        "player6_1/outdoor6__1_side_segment_processed.mp4",
        "trajectory/Cindy__trajectory/player6_1/outdoor6__1_side_segment_processed.mp4"
    ]
    
    # Also accept args
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        
    for f in files:
        if os.path.exists(f):
            convert_to_h264(f)
        else:
            # Search recursively if path is relative/uncertain
            pass # Simplified for now
