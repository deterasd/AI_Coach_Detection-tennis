"""
球進入方向控制測試程式
讓您可以輕鬆選擇球從左邊進入還是右邊進入
"""

# 從測試腳本資料夾匯入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_scripts'))
from trajector_processing_simple_test import simple_test_pipeline, interactive_setup

def ball_direction_control_demo():
    """球進入方向控制演示"""
    print("🎾 球進入方向控制演示")
    print("=" * 50)
    print("此程式讓您可以控制球是從左進入還是右進入")
    print()
    
    while True:
        print("🎯 請選擇球進入方向:")
        print("1. 🟢 右邊進入 (發球機在右側，球從右邊飛入)")
        print("2. 🔵 左邊進入 (發球機在左側，球從左邊飛入)")
        print("3. ❌ 退出程式")
        
        choice = input("\n請選擇 (1-3): ").strip()
        
        if choice == "1":
            ball_direction = "right"
            detection_area = "右邊上方2/3區域"
            print(f"\n✅ 已選擇: 右邊進入")
        elif choice == "2":
            ball_direction = "left"  
            detection_area = "左邊上方2/3區域"
            print(f"\n✅ 已選擇: 左邊進入")
        elif choice == "3":
            print("👋 再見！")
            return
        else:
            print("❌ 無效選擇，請重新輸入")
            continue
        
        print(f"🎯 偵測設定:")
        print(f"   球進入方向: {'右邊' if ball_direction == 'right' else '左邊'}")
        print(f"   偵測範圍: {detection_area}")
        print(f"   啟用球出場偵測: 是")
        print(f"   動態分割模式: 啟用")
        print(f"   出場等待時間: 1.5秒")
        
        # 設定信心度
        print(f"\n🔍 偵測信心度設定 (0.1-1.0):")
        print("- 較低值 (如0.3): 偵測更敏感，可能有誤判")
        print("- 較高值 (如0.7): 偵測更嚴格，可能遺漏")
        print("- 建議值: 0.5")
        
        confidence_input = input("\n請輸入信心度 (直接Enter使用0.5): ").strip()
        try:
            confidence_threshold = float(confidence_input) if confidence_input else 0.5
            confidence_threshold = max(0.1, min(1.0, confidence_threshold))  # 限制範圍
        except:
            confidence_threshold = 0.5
        
        print(f"\n🚀 開始執行分析...")
        print(f"   球進入方向: {'右邊' if ball_direction == 'right' else '左邊'}")
        print(f"   偵測信心度: {confidence_threshold}")
        
        confirm = input("\n確認執行？(y/n): ").lower().strip()
        if confirm == 'y':
            # 執行測試
            success = simple_test_pipeline(
                input_folder="input_videos",
                ball_direction=ball_direction,
                confidence_threshold=confidence_threshold
            )
            
            if success:
                print(f"\n🎉 分析完成！")
                print(f"📊 球進入方向設定: {'右邊' if ball_direction == 'right' else '左邊'}")
                print(f"📁 查看分割片段以確認偵測效果")
            else:
                print(f"\n😔 分析失敗，請檢查設定")
            
            input("\n按 Enter 返回主選單...")
        else:
            print("❌ 已取消執行")

if __name__ == "__main__":
    ball_direction_control_demo()