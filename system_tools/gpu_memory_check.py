"""
GPU 記憶體檢查和清理工具
"""
import torch
import gc

def check_gpu_status():
    """檢查 GPU 狀態"""
    print("🔍 檢查 GPU 狀態...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"📱 GPU 數量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # GB
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            
            print(f"🎮 GPU {i}: {props.name}")
            print(f"   總記憶體: {total_memory:.2f} GB")
            print(f"   已分配: {allocated:.2f} GB")
            print(f"   已緩存: {cached:.2f} GB")
            print(f"   可用: {total_memory - cached:.2f} GB")
    else:
        print("❌ CUDA 不可用")

def clear_all_gpu_memory():
    """清理所有 GPU 記憶體"""
    print("\n🧹 清理 GPU 記憶體...")
    
    if torch.cuda.is_available():
        # 清理所有 GPU
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        # 強制垃圾回收
        gc.collect()
        print("✅ GPU 記憶體清理完成")
    else:
        print("⚠️ 無 GPU 可清理")

if __name__ == "__main__":
    check_gpu_status()
    clear_all_gpu_memory()
    print("\n" + "="*50)
    print("清理後狀態:")
    check_gpu_status()