"""
GPU加速影片分割增強模組
利用FFmpeg的NVIDIA GPU加速功能大幅提升影片處理速度
"""

import subprocess
from pathlib import Path
import os

class GPUAcceleratedVideoProcessor:
    def __init__(self):
        self.ffmpeg_path = self._get_ffmpeg_path()
        self.gpu_available = self._check_gpu_support()
        
    def _get_ffmpeg_path(self):
        """獲取FFmpeg路徑"""
        # 檢查本地安裝
        local_ffmpeg = Path("tools/ffmpeg.exe")
        if local_ffmpeg.exists():
            return str(local_ffmpeg)
        return "ffmpeg"  # 系統安裝
    
    def _check_gpu_support(self):
        """檢查GPU支援"""
        try:
            result = subprocess.run([self.ffmpeg_path, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout.lower()
                return 'nvenc' in output and 'cuda' in output
        except:
            pass
        return False
    
    def get_gpu_info(self):
        """獲取GPU資訊"""
        if not self.gpu_available:
            return "GPU加速不可用"
        
        try:
            # 檢查NVIDIA-SMI
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'RTX' in line or 'GTX' in line:
                        return f"檢測到GPU: {line.strip()}"
            return "NVIDIA GPU已檢測到"
        except:
            return "GPU支援可用但無法獲取詳細資訊"
    
    def segment_video_gpu(self, input_path, output_path, start_time, duration):
        """使用GPU加速分割單個影片片段"""
        if not self.gpu_available:
            return self._segment_video_cpu(input_path, output_path, start_time, duration)
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-hwaccel', 'cuda',           # 硬體加速解碼
                '-hwaccel_output_format', 'cuda',  # GPU記憶體格式
                '-i', input_path,
                '-ss', str(start_time),       # 開始時間
                '-t', str(duration),          # 持續時間
                '-c:v', 'h264_nvenc',         # NVIDIA GPU編碼器
                '-preset', 'fast',            # 快速預設
                '-cq', '23',                  # 品質設定(lower = better)
                '-c:a', 'aac',                # 音訊編碼器
                '-b:a', '128k',               # 音訊位元率
                '-avoid_negative_ts', 'make_zero',
                '-y',                         # 覆蓋輸出檔案
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "GPU分割超時"
        except Exception as e:
            return False, f"GPU分割錯誤: {e}"
    
    def _segment_video_cpu(self, input_path, output_path, start_time, duration):
        """CPU備用分割方法"""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',            # CPU編碼器
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0, result.stderr
            
        except Exception as e:
            return False, f"CPU分割錯誤: {e}"
    
    def batch_segment_gpu(self, segments_info, progress_callback=None):
        """批量GPU加速分割"""
        results = []
        total = len(segments_info)
        
        for i, (input_path, output_path, start_time, duration) in enumerate(segments_info):
            if progress_callback:
                progress_callback(i, total, f"處理: {Path(output_path).name}")
            
            success, error = self.segment_video_gpu(input_path, output_path, start_time, duration)
            results.append({
                'output_path': output_path,
                'success': success,
                'error': error if not success else None
            })
        
        if progress_callback:
            progress_callback(total, total, "完成!")
        
        return results
    
    def get_video_info(self, video_path):
        """獲取影片資訊"""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            # FFmpeg輸出資訊通常在stderr中
            output = result.stderr
            
            info = {}
            for line in output.split('\n'):
                if 'Duration:' in line:
                    duration_part = line.split('Duration:')[1].split(',')[0].strip()
                    info['duration'] = duration_part
                elif 'Video:' in line:
                    info['video_codec'] = line.split('Video:')[1].split(',')[0].strip()
                elif 'fps' in line:
                    fps_part = [part for part in line.split() if 'fps' in part]
                    if fps_part:
                        info['fps'] = fps_part[0].replace('fps', '').strip()
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def estimate_processing_time(self, video_duration_seconds, num_segments):
        """估算處理時間"""
        if self.gpu_available:
            # GPU處理：每秒影片約需0.1-0.2秒處理時間
            base_time = video_duration_seconds * 0.15
        else:
            # CPU處理：每秒影片約需1-3秒處理時間
            base_time = video_duration_seconds * 2.0
        
        # 考慮分割數量的開銷
        segment_overhead = num_segments * 0.5
        
        total_time = base_time + segment_overhead
        return max(total_time, 10)  # 最少10秒
    
    def get_performance_report(self):
        """獲取性能報告"""
        report = {
            'ffmpeg_path': self.ffmpeg_path,
            'gpu_available': self.gpu_available,
            'gpu_info': self.get_gpu_info(),
            'estimated_speedup': '30-50x faster' if self.gpu_available else 'Standard speed'
        }
        return report

# 使用範例和測試
if __name__ == "__main__":
    processor = GPUAcceleratedVideoProcessor()
    print("GPU加速影片處理器初始化完成")
    print("性能報告:", processor.get_performance_report())