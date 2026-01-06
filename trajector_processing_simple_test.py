"""
簡易測試版本 - AI網球教練自動化分析系統
模擬正常的處理流程：
1. 使用者輸入身高和姓名
2. 在 trajectory/(姓名)__trajectory 資料夾中創建標準資料夾結構
3. 從 input_videos 讀取影片並進行完整分析
4. 將所有處理結果保存到對應的資料夾中
"""

import time
import numpy as np
import os
import subprocess
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import shutil
from datetime import datetime

# 強制重新載入模組以確保使用最新代碼
if 'trajector_processing_unified' in sys.modules:
    del sys.modules['trajector_processing_unified']

# 匯入整合處理函數 (從父目錄)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajector_processing_unified import processing_trajectory_unified

def get_user_info():
    """獲取使用者身高和姓名資訊"""
    print("👤 請輸入使用者資訊:")
    print("="*40)
    
    # 輸入姓名
    while True:
        name = input("請輸入姓名 (必填): ").strip()
        if name:
            # 移除可能導致檔案系統問題的字符
            invalid_chars = r'<>:"/\|?*'
            for char in invalid_chars:
                name = name.replace(char, '_')
            break
        else:
            print("❌ 姓名不能為空，請重新輸入")
    
    # 輸入身高
    while True:
        height_input = input("請輸入身高 (cm，例如: 175): ").strip()
        try:
            height = int(height_input)
            if 100 <= height <= 250:  # 合理的身高範圍
                break
            else:
                print("❌ 請輸入合理的身高 (100-250cm)")
        except ValueError:
            print("❌ 請輸入有效的數字")
    
    print(f"\n✅ 使用者資訊:")
    print(f"   姓名: {name}")
    print(f"   身高: {height} cm")
    
    return name, height

def check_and_install_ffmpeg():
    """檢查並安裝 FFmpeg"""
    try:
        # 檢查系統 PATH 中的 FFmpeg
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg 已安裝（系統 PATH）")
            return True
    except FileNotFoundError:
        pass
    
    # 檢查本地 tools 資料夾是否有 FFmpeg
    local_ffmpeg = Path("tools/ffmpeg.exe")
    if local_ffmpeg.exists():
        print("✅ FFmpeg 已安裝（本地版本）")
        print(f"📁 位置: {local_ffmpeg.absolute()}")
        try:
            # 測試本地 FFmpeg 是否可用
            result = subprocess.run([str(local_ffmpeg), '-version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ 本地 FFmpeg 測試通過")
                return True
        except Exception as e:
            print(f"⚠️ 本地 FFmpeg 測試失敗: {e}")
    
    print("❌ FFmpeg 未安裝，嘗試自動安裝...")
    try:
        # 嘗試使用 chocolatey 安裝（Windows）
        print("🔧 嘗試使用 Chocolatey 安裝 FFmpeg...")
        result = subprocess.run(['choco', 'install', 'ffmpeg', '-y'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg 安裝成功")
            return True
    except:
        pass
    
    print("⚠️ 無法自動安裝 FFmpeg")
    print("📋 請手動安裝 FFmpeg:")
    print("   1. 訪問: https://ffmpeg.org/download.html")
    print("   2. 下載 Windows 版本")
    print("   3. 解壓縮並添加到系統 PATH")
    print("   4. 或使用 Chocolatey: choco install ffmpeg")
    print("💡 或者：您的本地已有 tools/ffmpeg.exe，程式應該能正常運作")
    
    choice = input("\n是否跳過影片分割功能繼續執行？(y/n)1: ").lower().strip()
    return choice == 'y'

def create_trajectory_folders(name, height):
    """創建符合正常流程的資料夾結構"""
    # 主要軌跡資料夾
    base_trajectory_folder = Path("trajectory")
    user_folder = base_trajectory_folder / f"{name}__trajectory"
    
    print(f"📁 創建使用者資料夾: {user_folder}")
    
    # 創建標準的資料夾結構（模擬正常流程）
    folders = {
        "input_videos": user_folder,                                    # 輸入影片（放在根目錄）
        "synced_videos": user_folder,                                   # 同步後影片
        "segmented_videos": user_folder,                                # 分割片段
        "2d_trajectories": user_folder,                                 # 2D軌跡
        "processed_videos": user_folder,                                # 處理後影片
        "3d_trajectories": user_folder,                                 # 3D軌跡
        "analysis_results": user_folder,                                # 分析結果
        "final_reports": user_folder,                                   # 最終報告
        "logs": user_folder / "logs"                                    # 日誌檔案
    }
    
    # 創建主要資料夾
    user_folder.mkdir(parents=True, exist_ok=True)
    folders["logs"].mkdir(parents=True, exist_ok=True)
    
    # 創建使用者資訊檔案
    user_info = {
        "name": name,
        "height": height,
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    user_info_file = user_folder / "user_info.json"
    with open(user_info_file, 'w', encoding='utf-8') as f:
        json.dump(user_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 使用者資料夾創建完成: {user_folder}")
    print(f"📄 使用者資訊已保存: user_info.json")
    
    return user_folder, folders

def check_input_videos(input_folder="input_videos"):
    """檢查輸入影片是否存在"""
    input_path = Path(input_folder)
    
    print(f"🔍 檢查輸入資料夾: {input_path.absolute()}")
    
    if not input_path.exists():
        print(f"📁 創建輸入資料夾: {input_path}")
        input_path.mkdir(parents=True, exist_ok=True)
        print("📝 請將影片檔案放入以下資料夾:")
        print(f"   {input_path.absolute()}")
        print("📋 檔案命名規則:")
        print("   - 側面影片: 包含 'side' 或 '側面' 的檔名")
        print("   - 45度影片: 包含 '45' 或 '角度' 的檔名")
        print("   - 例如: tennis_side.mp4, tennis_45.mp4")
        return None, None
    
    # 尋找影片檔案
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_videos = []
    
    for ext in video_extensions:
        all_videos.extend(input_path.glob(f"*{ext}"))
    
    if not all_videos:
        print("❌ 沒有找到影片檔案")
        print(f"📁 請將影片檔案放入: {input_path.absolute()}")
        return None, None
    
    # 智能識別側面和45度影片
    side_video = None
    deg45_video = None
    
    for video in all_videos:
        video_name_lower = video.name.lower()
        if any(keyword in video_name_lower for keyword in ['side', '側面', 'lateral']):
            side_video = video
        elif any(keyword in video_name_lower for keyword in ['45', '角度', 'angle']):
            deg45_video = video
    
    # 如果沒有找到特定關鍵字，按字母順序分配
    if not side_video or not deg45_video:
        sorted_videos = sorted(all_videos)
        if len(sorted_videos) >= 2:
            side_video = sorted_videos[0]
            deg45_video = sorted_videos[1]
            print("⚠️ 無法根據檔名自動識別，按字母順序分配:")
        elif len(sorted_videos) == 1:
            print("❌ 只找到一個影片檔案，需要兩個角度的影片")
            return None, None
    
    if side_video and deg45_video:
        print(f"✅ 找到影片檔案:")
        print(f"   側面影片: {side_video.name}")
        print(f"   45度影片: {deg45_video.name}")
        return str(side_video), str(deg45_video)
    else:
        print("❌ 無法找到足夠的影片檔案")
        return None, None

def copy_input_videos(side_video, deg45_video, user_folder, name):
    """複製輸入影片到使用者資料夾，使用正常流程的命名方式"""
    user_folder = Path(user_folder)
    
    # 使用正常流程的命名方式: (姓名)__(編號)_side.mp4 和 (姓名)__(編號)_45.mp4
    side_dest = user_folder / f"{name}__1_side.mp4"
    deg45_dest = user_folder / f"{name}__1_45.mp4"
    
    shutil.copy2(side_video, side_dest)
    shutil.copy2(deg45_video, deg45_dest)
    
    print(f"📋 輸入影片已複製並重新命名:")
    print(f"   側面影片: {side_dest.name}")
    print(f"   45度影片: {deg45_dest.name}")
    
    return str(side_dest), str(deg45_dest)

def create_readme_file(user_folder, name, height, side_video, deg45_video, ball_direction):
    """創建說明檔案"""
    readme_content = f"""
# AI網球教練分析報告 - {name}
生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 使用者資訊
- 姓名: {name}
- 身高: {height} cm
- 資料夾: {name}__trajectory

## 輸入資料
- 側面影片: {Path(side_video).name}
- 45度影片: {Path(deg45_video).name}
- 球進入方向: {'右邊' if ball_direction == 'right' else '左邊'}

## 處理流程
1. 影片時間同步
2. 智能分割（自動偵測擊球時機）
3. 2D軌跡提取和平滑處理
4. 影片物件偵測處理
5. 3D軌跡重建
6. 動作分析和比較
7. AI生成改進建議

## 生成的檔案
所有處理結果都保存在此資料夾中，包括：
- 原始和處理後的影片檔案
- 2D和3D軌跡JSON檔案
- KNN分析結果
- GPT生成的建議報告
- 執行日誌和錯誤記錄

## 注意事項
- 此資料夾模擬正常的processing流程結果
- 檔案命名遵循標準格式: {name}__編號_角度.副檔名
- 可以直接在 drawing_3D_three_js.html 中載入查看
"""
    
    readme_path = user_folder / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📄 說明檔案已創建: README.md")

def simple_test_pipeline(input_folder="input_videos", 
                         ball_direction="right", 
                         confidence_threshold=0.5):
    """模擬正常流程的測試流程"""
    
    print("🎾 AI網球教練 - 正常流程模擬版本")
    print("=" * 60)
    
    # 步驟1: 獲取使用者資訊
    print("\n👤 步驟1: 輸入使用者資訊...")
    name, height = get_user_info()
    
    # 步驟2: 檢查輸入影片
    print("\n📹 步驟2: 檢查輸入影片...")
    side_video, deg45_video = check_input_videos(input_folder)
    
    if not side_video or not deg45_video:
        print("\n❌ 測試終止：請先準備影片檔案")
        print(f"📁 將影片放入: {Path(input_folder).absolute()}")
        print("🔄 然後重新執行此程式")
        input("\n按 Enter 結束...")
        return False
    
    # 步驟3: 創建使用者資料夾
    print(f"\n📁 步驟3: 創建 {name} 的軌跡資料夾...")
    user_folder, folders = create_trajectory_folders(name, height)
    print(f"📂 使用者資料夾: {user_folder}")
    
    # 步驟4: 複製並重命名輸入影片
    print("\n📋 步驟4: 複製輸入影片到使用者資料夾...")
    side_video_copy, deg45_video_copy = copy_input_videos(
        side_video, deg45_video, user_folder, name
    )
    
    # 步驟5: 創建說明檔案
    create_readme_file(user_folder, name, height, side_video, deg45_video, ball_direction)
    
    # 步驟6: 檢查 FFmpeg
    print(f"\n🔧 步驟6: 檢查 FFmpeg...")
    ffmpeg_available = check_and_install_ffmpeg()
    
    # 步驟7: 設定參數
    print(f"\n⚙️ 步驟7: 設定分析參數...")
    print(f"   使用者: {name} ({height} cm)")
    print(f"   球進入方向: {'右邊' if ball_direction == 'right' else '左邊'}")
    print(f"   偵測信心度: {confidence_threshold}")
    print(f"   影片分割功能: {'啟用' if ffmpeg_available else '停用（缺少FFmpeg）'}")
    print(f"   輸出資料夾: trajectory/{name}__trajectory")
    
    try:
        # 步驟7: 載入AI模型
        print("\n🤖 步驟7: 載入AI模型...")
        
        # 投影矩陣設定
        P1 = np.array([
             [  916.626242,     0.000000,   960.250417,     0.000000],
             [    0.000000,   921.951283,   523.154606,     0.000000],
             [    0.000000,     0.000000,     1.000000,     0.000000],
        ])

        P2 = np.array([
            [  782.909772,   -18.152980,  1066.677600, -255341.954492],
            [  -25.104948,   925.678666,   514.730223, 46851.486878],
            [   -0.122625,     0.020539,     0.992241,    90.876653], # 0.0047 -> -0.0047
        ])
        
        # 載入YOLO模型
        print("📦 載入 YOLO 模型...")
        yolo_pose_model = YOLO('model/yolov8n-pose.pt')
        yolo_tennis_ball_model = YOLO('model/tennisball_OD_v1.pt')
        yolo_paddle_model = YOLO('model/tennispaddle.pt')  # 新增：載入球拍模型
        
        # GPU加速（安全檢查）
        print("🔍 檢查 GPU 可用性...")
        try:
            import torch
            if torch.cuda.is_available():
                # 檢查 GPU 記憶體
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   總記憶體: {total_memory:.2f} GB")
                
                # 清理現有的 GPU 記憶體
                torch.cuda.empty_cache()
                
                # 謹慎地移動模型到 GPU
                if total_memory >= 4.0:  # 至少需要 4GB
                    try:
                        yolo_pose_model.model.to('cuda')
                        yolo_tennis_ball_model.model.to('cuda')
                        yolo_paddle_model.model.to('cuda')  # 新增：移動球拍模型到 GPU
                        print("⚡ GPU 加速已啟用")
                    except Exception as gpu_error:
                        print(f"⚠️ GPU 設置失敗: {gpu_error}")
                        print("💻 回退到 CPU 模式")
                        yolo_pose_model.model.to('cpu')
                        yolo_tennis_ball_model.model.to('cpu')
                        yolo_paddle_model.model.to('cpu')  # 新增：回退到 CPU
                else:
                    print("⚠️ GPU 記憶體不足 (需要至少 4GB)")
                    print("💻 使用 CPU 模式")
                    yolo_pose_model.model.to('cpu')
                    yolo_tennis_ball_model.model.to('cpu')
                    yolo_paddle_model.model.to('cpu')  # 新增
            else:
                print("💻 GPU 不可用，使用 CPU 模式")
                yolo_pose_model.model.to('cpu')
                yolo_tennis_ball_model.model.to('cpu')
                yolo_paddle_model.model.to('cpu')  # 新增
        except Exception as e:
            print(f"⚠️ GPU 檢查失敗: {e}")
            print("💻 使用 CPU 模式")
            yolo_pose_model.model.to('cpu')
            yolo_tennis_ball_model.model.to('cpu')
            yolo_paddle_model.model.to('cpu')  # 新增
        
        # KNN資料集
        knn_dataset = 'knn_dataset.json'
        
        # 步驟8: 執行完整分析流程
        print("\n🚀 步驟8: 開始完整分析流程...")
        print("⏳ 這可能需要幾分鐘時間，請耐心等待...")
        print(f"📁 所有結果將保存到: {user_folder}")
        
        # 根據 FFmpeg 可用性決定是否分割影片
        segment_videos = ffmpeg_available
        
        # 強制啟用分割功能，根據您的要求
        print(f"\n📹 影片分割設定:")
        print(f"   球進入偵測範圍: 右邊上方2/3區域")
        print(f"   啟用球出場偵測: 是")
        print(f"   動態分割模式: 啟用")
        print(f"   出場等待時間: 1.5秒")
        
        # 確保分割功能啟用
        segment_videos = True  # 強制啟用
        
        if not ffmpeg_available:
            print("⚠️ FFmpeg 系統PATH檢查失敗，但將嘗試使用本地版本進行分割")
        else:
            print("✅ 影片分割功能已啟用")
        
        # 使用新的統一處理函數
        success = processing_trajectory_unified(
            P1=P1, 
            P2=P2, 
            yolo_pose_model=yolo_pose_model, 
            yolo_tennis_ball_model=yolo_tennis_ball_model,
            yolo_paddle_model=yolo_paddle_model,  # 新增：傳遞球拍模型
            video_side=side_video_copy, 
            video_45=deg45_video_copy, 
            knn_dataset=knn_dataset,
            name=name,  # 傳入使用者姓名
            ball_entry_direction=ball_direction,
            confidence_threshold=confidence_threshold,
            output_folder=str(user_folder),  # 使用使用者資料夾
            segment_videos=segment_videos   # 根據 FFmpeg 可用性決定
        )
        
        if success:
            print("\n🎉 分析流程完成！")
            print(f"📂 所有結果已保存到: {user_folder}")
            print("\n📋 生成的檔案:")
            
            # 檢查使用者資料夾中的檔案
            if user_folder.exists():
                all_files = list(user_folder.glob("*"))
                video_files = [f for f in all_files if f.suffix.lower() in ['.mp4', '.avi', '.mov']]
                json_files = [f for f in all_files if f.suffix.lower() == '.json']
                other_files = [f for f in all_files if f not in video_files + json_files and f.is_file()]
                
                print(f"   📹 影片檔案: {len(video_files)} 個")
                for video in video_files:
                    print(f"      - {video.name}")
                
                print(f"   📊 軌跡/分析檔案: {len(json_files)} 個")
                for json_file in json_files:
                    print(f"      - {json_file.name}")
                
                if other_files:
                    print(f"   📄 其他檔案: {len(other_files)} 個")
                    for other in other_files:
                        print(f"      - {other.name}")
            
            print(f"\n📄 詳細說明請查看: {user_folder / 'README.md'}")
            print(f"🌐 可以在 drawing_3D_three_js.html 中載入 {name} 的分析結果")
            return True
            
        else:
            print("\n❌ 分析流程失敗")
            return False
            
    except Exception as e:
        print(f"\n💥 執行過程發生錯誤: {e}")
        print("📝 錯誤詳情已記錄在logs資料夾中")
        
        # 記錄錯誤到日誌
        error_log = folders["logs"] / "error.log"
        with open(error_log, 'w', encoding='utf-8') as f:
            f.write(f"錯誤時間: {datetime.now()}\n")
            f.write(f"使用者: {name} ({height} cm)\n")
            f.write(f"錯誤訊息: {str(e)}\n")
            f.write(f"輸入影片: {side_video}, {deg45_video}\n")
        
        return False

def load_config():
    """載入設定檔案"""
    config_file = Path("config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("✅ 已載入設定檔案")
            return config
        except Exception as e:
            print(f"⚠️ 設定檔案載入失敗: {e}")
    
    # 創建預設設定
    default_config = {
        "ball_direction": "right",
        "confidence_threshold": 0.5,
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    }
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print("✅ 已創建預設設定檔案")
    except Exception as e:
        print(f"⚠️ 無法創建設定檔案: {e}")
    
    return default_config

def save_config(config):
    """保存設定檔案"""
    config_file = Path("config.json")
    config["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("✅ 設定已保存")
        return True
    except Exception as e:
        print(f"❌ 設定保存失敗: {e}")
        return False

def interactive_setup():
    """載入設定並顯示"""
    print("🎾 AI網球教練 - 正常流程模擬版本")
    print("=" * 60)
    print("📋 此程式會模擬正常的處理流程，在 trajectory/(姓名)__trajectory 中保存所有結果")
    print()
    
    # 載入現有設定
    config = load_config()
    ball_direction = config.get("ball_direction", "right")
    confidence_threshold = config.get("confidence_threshold", 0.5)
    
    print("📁 當前設定:")
    print(f"   球進入方向: {'右邊' if ball_direction == 'right' else '左邊'}")
    print(f"   偵測信心度: {confidence_threshold}")
    print("� 如需修改設定，請編輯 config.json 檔案")
    print()
    
    return ball_direction, confidence_threshold

if __name__ == "__main__":
    print("🎾 AI網球教練 - 正常流程模擬啟動")
    print("=" * 50)
    
    # 互動式設定
    ball_direction, confidence_threshold = interactive_setup()
    
    print("\n🚀 開始測試流程...")
    input("按 Enter 繼續...")
    
    # 執行測試
    success = simple_test_pipeline(
        input_folder="input_videos",
        ball_direction=ball_direction,
        confidence_threshold=confidence_threshold
    )
    
    if success:
        print("\n✨ 恭喜！正常流程模擬成功！")
        print("📊 現在可以在 trajectory/ 資料夾中查看結果")
        print("🔍 特別注意分割的影片片段和最終的AI建議")
        print("🌐 可以在 drawing_3D_three_js.html 中載入並查看3D軌跡")
    else:
        print("\n😔 處理過程中遇到問題")
        print("🔧 請檢查:")
        print("   1. 影片檔案格式是否正確")
        print("   2. 模型檔案是否存在")
        print("   3. 網路連接是否正常（GPT功能需要）")
    
    print(f"\n📁 結果資料夾位置: trajectory/(姓名)__trajectory/")
    input("\n按 Enter 結束程式...")
    