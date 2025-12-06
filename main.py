import time
import json
import asyncio
from pathlib import Path
from enum import Enum
import numpy as np
from typing import Optional  
import pygame
import os
import sys
from googletrans import Translator
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO


from trajector_processing import processing_trajectory
print("processing_trajectory:", processing_trajectory)
from trajectory_gpt_overall_feedback import find_and_format_feedback_jsons, conclude

# ------------------------------
# Calibration Matrices
# ------------------------------
P1 = np.array([
    [  613.902729,     0.000000,   638.203915,     0.000000],
    [    0.000000,   617.251817,   364.556522,     0.000000],
    [    0.000000,     0.000000,     1.000000,     0.000000],
    ])
P2 = np.array([
     [  616.071259,     7.588060,   617.077541, 154727.853092],
    [   -0.773895,   591.674918,   358.669757, -16209.393573],
    [    0.038272,    -0.010667,     0.999210,   -68.044380],
])
"""
P1 = np.array([
    [  338.943814,     0.000000,   689.299962,     0.000000],
    [    0.000000,   343.163221,   551.893014,     0.000000],
    [    0.000000,     0.000000,     1.000000,     0.000000],
])
P2 = np.array([
    [  234.305465,   221.589827,   888.881308, -63641.642298],
    [ -312.141214,   760.908921,   381.007762, 519415.743557],
    [   -0.485477,     0.205747,     0.849694,   768.163985],
])

P1 = np.array([
    [  682.525930,     0.000000,   637.087464,     0.000000],
    [    0.000000,   684.519186,   360.040032,     0.000000],
    [    0.000000,     0.000000,     1.000000,     0.000000],
])

P2 = np.array([
    [  572.714085,    27.508121,   700.861208, -159410.297486],
    [  -27.187871,   663.411218,   332.613656, 51347.982959],
    [   -0.102197,     0.053920,     0.993302,    73.630082],
])
"""
"""
P1 = np.array([
    [ 2199.047061,     0.000000,  2764.837466,     0.000000],
    [    0.000000,  2208.700298,  2437.493986,     0.000000],
    [    0.000000,     0.000000,     1.000000,     0.000000],
])

P2 = np.array([
    [ 1602.619612,   -35.094977,  3210.224680, -683196.019800],
    [ -426.660513,  2184.000907,  2380.884968, 80418.748408],
    [   -0.188707,    -0.001726,     0.982032,    43.029597],
])
"""
# ------------------------------
# Global Variables & Queue
# ------------------------------
current_user_folder: Optional[Path] = None
current_user_name: Optional[str] = None

yolo_pose_model: Optional[YOLO] = None
yolo_tennis_ball_model: Optional[YOLO] = None
paddle_model: Optional[YOLO] = None  # 👈 0923新增球拍模型

# 全域隊列，用來儲存軌跡處理任務
trajectory_queue: asyncio.Queue = asyncio.Queue()
active_task_count = 0
finished_task_count = 0


# ------------------------------
# FastAPI App Initialization
# ------------------------------
app = FastAPI(title="GoPro Controller API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------
# Enums & Data Models
# ------------------------------
class DominantHand(str, Enum):
    right = "right"
    left = "left"

class UserData(BaseModel):
    name: str
    height: float
    dominant_hand: str

# ------------------------------
# Utility Functions
# ------------------------------
def find_next_trajectory_number(base_folder: Path) -> int:
    """
    根據 base_folder 中現有的軌跡資料夾，返回下一個可用的編號。
    """
    try:
        if not base_folder.exists():
            return 1

        max_number = 0
        for folder in base_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("trajectory__"):
                try:
                    number = int(folder.name.split("__")[-1])
                    max_number = max(max_number, number)
                except (ValueError, IndexError):
                    continue
        return max_number + 1
    except Exception as e:
        print(f"Error in find_next_trajectory_number: {str(e)}")
        return 1

def play_sound():
    # 初始化pygame混音器
    pygame.mixer.init()
    
    # 設定預設音效文件
    # 你可以替換這個路徑為你自己的音效文件
    sound_file = "tool/sound.mp3"  # 預設音效檔案名稱
    
    # 檢查是否有通過命令列提供音效檔案路徑
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        sound_file = sys.argv[1]
    
    # 檢查檔案是否存在
    if not os.path.exists(sound_file):
        print(f"找不到音效文件: {sound_file}")
        print("請確保音效文件存在，或者通過命令列參數提供正確的路徑")
        print("用法: python script.py [音效文件路徑]")
        time.sleep(3)  # 讓用戶有時間閱讀錯誤信息
        return
    
    try:
        # 載入並播放音效
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
        
        # 等待音效播放完畢
        duration = sound.get_length()
        time.sleep(duration)
        
        print(f"已播放音效: {sound_file}")
        
    except Exception as e:
        print(f"播放音效時發生錯誤: {e}")
        time.sleep(3)  # 讓用戶有時間閱讀錯誤信息

# ------------------------------
# Task Worker for Sequential Processing
# ------------------------------
async def trajectory_worker():
    """
    持續監聽 trajectory_queue，逐一處理軌跡任務。
    每次取出一個任務後，執行 processing_trajectory，
    任務完成後更新 active_task_count 與 finished_task_count。
    """
    global active_task_count, finished_task_count
    while True:
        task_args = await trajectory_queue.get()
        active_task_count += 1  # 任務開始執行
        try:
            # 解包任務參數
            P1_, P2_, pose_model, ball_model, paddle_model, side_video, video_45, knn_dataset = task_args
            print("開始處理軌跡任務...")
            await asyncio.to_thread(
                processing_trajectory,
                P1_, P2_, pose_model, ball_model,paddle_model,
                side_video, video_45, knn_dataset
            )# 👈0923 加入 paddle 模型
            print("軌跡處理完成，任務結束並釋放資源。")
        except Exception as e:
            print(f"處理軌跡任務時發生錯誤: {str(e)}")
        finally:
            active_task_count -= 1  # 任務結束執行
            finished_task_count += 1  # 記錄完成任務數
            trajectory_queue.task_done()
            print("任務已結束，等待下一個任務...")


async def post_gopro(session: aiohttp.ClientSession, url: str, data: Optional[dict] = None) -> dict:
    """
    封裝對 GoPro API 發送 POST 請求。
    """
    try:
        if isinstance(data, dict):
            form = aiohttp.FormData()
            for key, value in data.items():
                form.add_field(key, str(value))
            async with session.post(url, data=form) as response:
                return await response.json()
        elif data is not None:
            async with session.post(url, data=data) as response:
                return await response.json()
        else:
            async with session.post(url) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}

async def wait_for_file_ready(file_path: str, timeout: int = 120, check_interval: int = 2) -> bool:
    """
    檢查檔案是否完成寫入，依據檔案大小是否穩定來確認。
    """
    path = Path(file_path)
    if not path.exists():
        return False

    start_time = time.time()
    last_size = path.stat().st_size

    while True:
        if time.time() - start_time > timeout:
            print(f"Timeout waiting for {file_path} to complete")
            return path.exists()
        
        await asyncio.sleep(check_interval)

        if not path.exists():
            return False

        current_size = path.stat().st_size
        if current_size == last_size:
            await asyncio.sleep(check_interval)
            if path.exists() and path.stat().st_size == current_size:
                print(f"File {file_path} is ready with size {current_size} bytes")
                return True
        last_size = current_size
        print(f"File {file_path} still being written, current size: {current_size} bytes")

# ------------------------------
# Task Worker for Sequential Processing
# ------------------------------
async def trajectory_worker():
    
    #持續監聽 trajectory_queue，逐一處理軌跡任務。
    #每次取出一個任務後，執行 processing_trajectory，完成後自動結束該任務。
    
    while True:
        # 等待新的任務進入隊列
        task_args = await trajectory_queue.get()
        try:
            # 解包任務參數
            P1_, P2_, pose_model, ball_model,paddle_model, side_video, video_45, knn_dataset = task_args
            print("開始處理軌跡任務...")
            print("即將執行 processing_trajectory")
            await asyncio.to_thread(
                processing_trajectory,
                P1_, P2_, pose_model, ball_model,paddle_model,
                side_video, video_45, knn_dataset
            )# 👈0923 加入 paddle 模型
            print("已執行 processing_trajectory")
            print("軌跡處理完成，任務結束並釋放資源。")
        except Exception as e:
            print(f"處理軌跡任務時發生錯誤: {str(e)}")
        finally:
            trajectory_queue.task_done()
            print("任務已結束，等待下一個任務...")

# ------------------------------
# Application Startup Event
# ------------------------------
@app.on_event("startup")
async def startup_event():
    """
    伺服器啟動時載入 YOLO 模型，並啟動軌跡處理工作者。
    """
    global yolo_pose_model, yolo_tennis_ball_model,paddle_model
    print("正在載入 YOLO 模型...")
    try:
        yolo_pose_model = YOLO('model/yolov8n-pose.pt')
        yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
        paddle_model = YOLO('model/best-paddlekeypoint.pt')  # 👈 新增球拍模型
        #yolo_paddle_model = YOLO('model/best.pt')
        #print("球拍模型載入完成:", yolo_paddle_model)

        print("YOLO 模型載入完成!")
    except Exception as e:
        print(f"模型載入失敗: {str(e)}")
        raise e

    # 啟動軌跡處理工作者，確保任務依序處理
    asyncio.create_task(trajectory_worker())

# ------------------------------
# API Endpoints
# ------------------------------
@app.get("/model_status")
async def check_model_status():
    """
    回傳 YOLO 模型是否已成功載入。
    """
    return {
        "pose_model_loaded": yolo_pose_model is not None,
        "tennis_ball_model_loaded": yolo_tennis_ball_model is not None,
        "paddle_model_loaded": paddle_model is not None  # 👈 0923新增球拍模型
    }

@app.get("/input_data")
async def input_user_data(
    name: str,
    height: float,
    dominant_hand: int  # 0為左手，1為右手
):
    """
    接收使用者資料，建立使用者專屬資料夾與 JSON 記錄。
    """
    start_time = time.time()
    try:
        if dominant_hand not in [0, 1]:
            raise HTTPException(
                status_code=400,
                detail="dominant_hand must be 0 (left) or 1 (right)"
            )
        
        hand = "left" if dominant_hand == 0 else "right"
        global current_user_folder, current_user_name
        current_user_name = name

        # 建立使用者資料夾
        current_user_folder = Path(f"trajectory/{name}__trajectory")
        current_user_folder.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "name": name,
            "height": height,
            "hand": hand,
            "timestamp": time.strftime("%Y_%m_%d_%H_%M_%S"),
            "file_path": str(current_user_folder)
        }

        file_path = f"play_records/{name}_{str(height).replace('.0','')}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        execution_time = time.time() - start_time
        return {
            "message": "200 success",
            "data": data_to_save,
            "file_path": file_path,
            "execution_time": f"{execution_time:.2f} seconds"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": f"{execution_time:.2f} seconds"
            }
        )

@app.get("/gpt_response")
async def gpt_response():
    """
    檢查使用者資料，執行 GPT 回饋處理，並回傳最終結論。
    """
    global current_user_folder, current_user_name
    if not current_user_folder or not current_user_name:
        raise HTTPException(
            status_code=400,
            detail="User information not found. Please call /input_data first."
        )
    
    try:
        gpt_single_results = await asyncio.to_thread(
            find_and_format_feedback_jsons,
            current_user_folder
        )
        final_conclusion = await asyncio.to_thread(
            conclude,
            gpt_single_results
        )
        print('------------------')
        print(current_user_folder)
        print(gpt_single_results)
        print(final_conclusion)
        print('------------------')
        return {
            "status": "success",
            "user_name": current_user_name,
            "conclusion": final_conclusion
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "user_name": current_user_name,
                "user_folder": str(current_user_folder)
            }
        )

@app.get("/take_photo")
async def take_photo():
    """
    同時向兩台 GoPro 發送拍照請求。
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/take_photo"),
            post_gopro(session, "http://localhost:9436/take_photo")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/start_recording")
async def start_recording():
    asyncio.get_event_loop().run_in_executor(None, play_sound)
    """
    同時向兩台 GoPro 發送開始錄影請求。
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/start_recording"),
            post_gopro(session, "http://localhost:9436/start_recording")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/stop_recording")
async def stop_recording():
    """
    同時向兩台 GoPro 發送停止錄影請求。
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/stop_recording"),
            post_gopro(session, "http://localhost:9436/stop_recording")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/download")
async def download(background_tasks: BackgroundTasks):
    """
    同時對兩台 GoPro 停止錄影並下載影片，建立軌跡資料夾，檢查檔案是否準備好後，
    將軌跡處理任務加入隊列，由工作者依序處理，避免同時大量運算。
    """
    global current_user_folder, current_user_name
    if not current_user_folder or not current_user_name:
        raise HTTPException(
            status_code=400,
            detail="User information not found. Please call /input_data first."
        )
    
    base_folder = Path(current_user_folder)
    base_folder.mkdir(parents=True, exist_ok=True)
    
    next_number = find_next_trajectory_number(base_folder)
    trajectory_folder = base_folder / f"trajectory__{next_number}"
    trajectory_folder.mkdir(parents=True, exist_ok=True)
    
    form_data = {
        "user_name": current_user_name,
        "user_folder": str(current_user_folder),
        "trajectory_folder": str(trajectory_folder),
        "next_number": str(next_number)
    }
    
    print(f"Sending data to GoPros: {form_data}")
    
    async with aiohttp.ClientSession() as session:
        try:
            results = await asyncio.gather(
                post_gopro(session, "http://localhost:3253/download", form_data),
                post_gopro(session, "http://localhost:9436/download", form_data)
            )
            gopro1_result, gopro2_result = results[0], results[1]
            print(f"GoPro 1 response: {gopro1_result}")
            print(f"GoPro 2 response: {gopro2_result}")
            
            side_video_path = gopro1_result.get("video_path")
            video_45_path = gopro2_result.get("video_path")
            video_files_ready = False
            
            if (isinstance(gopro1_result, dict) and isinstance(gopro2_result, dict) and
                "download_status" in gopro1_result and "download_status" in gopro2_result):
                if (side_video_path and video_45_path and 
                    Path(side_video_path).exists() and Path(video_45_path).exists()):
                    
                    side_video_ready = await wait_for_file_ready(side_video_path)
                    video_45_ready = await wait_for_file_ready(video_45_path)
                    
                    if side_video_ready and video_45_ready:
                        video_files_ready = True
                        print("Both videos confirmed ready")
                        # 將處理任務加入隊列，等待工作者依序處理
                        knn_dataset_path = 'knn_dataset.json'
                        if not Path(knn_dataset_path).exists():
                            print(f"knn_dataset.json not found at: {knn_dataset_path}")
                            raise HTTPException(
                                status_code=500,
                                detail=f"knn_dataset.json not found at: {knn_dataset_path}"
                            )
                        with open(knn_dataset_path, 'r', encoding='utf-8') as f:
                            knn_data = json.load(f)
                        # 將處理任務加入隊列，等待工作者依序處理
                        await trajectory_queue.put(
                            (P1, P2, yolo_pose_model, yolo_tennis_ball_model, paddle_model,
                            side_video_path, video_45_path, knn_data)
                        )
                        #await trajectory_queue.put(
                         #   (P1, P2, yolo_pose_model, yolo_tennis_ball_model,paddle_model,
                          #   side_video_path, video_45_path, 'knn_dataset.json') 
                        #)# 👈 0923加入 paddle 模型
                    else:
                        if not side_video_ready:
                            print(f"Side video not fully written at: {side_video_path}")
                        if not video_45_ready:
                            print(f"45-degree video not fully written at: {video_45_path}")
                else:
                    if not side_video_path or not Path(side_video_path).exists():
                        print(f"Side video not found at: {side_video_path}")
                    if not video_45_path or not Path(video_45_path).exists():
                        print(f"45-degree video not found at: {video_45_path}")
            
            response_data = {
                "gopro1": gopro1_result,
                "gopro2": gopro2_result,
                "user_name": current_user_name,
                "user_folder": str(current_user_folder),
                "trajectory_folder": str(trajectory_folder),
                "videos_ready": video_files_ready,
                "form_data_sent": form_data
            }
            
            if video_files_ready:
                response_data["side_video"] = side_video_path
                response_data["video_45"] = video_45_path
            
            return response_data
        except Exception as e:
            print(f"Error in download: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "user_name": current_user_name,
                    "user_folder": str(current_user_folder),
                    "trajectory_folder": str(trajectory_folder)
                }
            )

@app.get("/translate")
async def translate(text: str = Query(..., description="要翻譯的文字")):
    """
    接收文字並將其翻譯成英文。
    """
    try:
        translator = Translator()
        result = await translator.translate(text, dest="en")
        return {
            "status": "success",
            "original_text": text,
            "translated_text": result.text
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"翻譯失敗: {str(e)}",
                "original_text": text
            }
        )
# ------------------------------
# Main Entry Point
# ------------------------------
if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)

