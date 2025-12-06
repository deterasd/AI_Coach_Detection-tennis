#!/usr/bin/env python3
"""
掃描網路上的 GoPro 裝置
"""
import socket
import threading
import time

def scan_port(ip, port, timeout=1):
    """掃描特定 IP:Port 是否開放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def scan_gopro_network():
    """掃描可能的 GoPro IP 範圍"""
    print("掃描 GoPro 網路連接...")
    
    # GoPro 常用的 IP 範圍
    ip_ranges = [
        "172.22.153.",  # 從錯誤信息中看到的
        "172.24.136.",  # 從錯誤信息中看到的
        "172.20.151.",  # 常見的 GoPro IP 範圍
        "172.21.111.",
        "172.23.171.",
        "10.5.5.",      # 另一個常見範圍
    ]
    
    found_devices = []
    
    for ip_range in ip_ranges:
        print(f"掃描 {ip_range}x:8080...")
        for i in [1, 51, 100]:  # 常見的最後一位數字
            ip = f"{ip_range}{i}"
            if scan_port(ip, 8080, timeout=2):
                print(f"✅ 找到活躍的裝置: {ip}:8080")
                found_devices.append(ip)
    
    return found_devices

def check_physical_connection():
    """檢查實體連接狀態"""
    import subprocess
    
    try:
        # 檢查 USB 網路介面
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        
        # 尋找可能的 GoPro 網路介面
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'usb' in line.lower() or 'rndis' in line.lower():
                print(f"找到可能的 USB 網路介面:")
                # 顯示這個介面的詳細信息
                for j in range(i, min(i+10, len(lines))):
                    if lines[j].strip():
                        print(f"  {lines[j]}")
                    if 'IPv4' in lines[j]:
                        break
        
    except Exception as e:
        print(f"檢查網路介面失敗: {e}")

if __name__ == "__main__":
    print("=== GoPro 網路連接診斷 ===")
    print()
    
    print("1. 檢查實體網路介面:")
    check_physical_connection()
    print()
    
    print("2. 掃描 GoPro 網路連接:")
    devices = scan_gopro_network()
    print()
    
    if devices:
        print(f"找到 {len(devices)} 個可能的 GoPro 裝置:")
        for device in devices:
            print(f"  http://{device}:8080")
        print()
        print("你可以在瀏覽器中測試這些網址")
    else:
        print("未找到活躍的 GoPro 裝置")
        print()
        print("可能的問題:")
        print("1. GoPro 未正確建立 USB 網路連接")
        print("2. Windows 網路介面卡驅動問題") 
        print("3. GoPro USB 模式未正確啟動")