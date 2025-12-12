"""
雙目3D重建驗證 - 配置管理模組
統一管理所有驗證步驟的閾值和參數
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ValidationConfig:
    """
    驗證配置類別
    
    包含所有驗證步驟的閾值和參數設定
    可從 JSON 檔案載入或保存配置
    """
    
    # ========== Step 1: 重投影誤差 ==========
    reprojection_outlier_threshold: float = 20.0  # px - 異常值閾值
    reprojection_95th_threshold: float = 15.0     # px - 95分位數閾值
    reprojection_excellent: float = 2.0           # px - 優秀標準
    reprojection_good: float = 5.0                # px - 良好標準
    reprojection_acceptable: float = 10.0         # px - 可接受標準
    side_camera_distortion: List[float] = field(default_factory=list)   # k1~k3, p1, p2
    camera_45_distortion: List[float] = field(default_factory=list)
    
    # ========== Step 2: 骨骼一致性 ==========
    bone_cv_excellent: float = 2.0        # % - 變異係數優秀標準
    bone_cv_good: float = 5.0             # % - 變異係數良好標準
    bone_cv_acceptable: float = 10.0      # % - 變異係數可接受標準
    
    bone_spike_ratio: float = 0.05        # 骨長跳動比例閾值 (5%)
    bone_spike_min_mm: float = 30.0       # mm - 最小跳動閾值
    
    bone_symmetry_excellent: float = 2.0  # % - 對稱性優秀標準
    bone_symmetry_acceptable: float = 5.0 # % - 對稱性可接受標準
    
    bone_correction_threshold: float = 0.10  # 骨長修正閾值 (10%)
    
    # ========== Step 3: 深度合理性 ==========
    depth_min_mm: float = 0.0              # mm - 預設最小深度（僅檢查正值）
    depth_max_mm: float = 10000.0          # mm - 預設最大深度（10 公尺）
    
    depth_cv_excellent: float = 2.0        # % - 深度CV優秀標準
    depth_cv_good: float = 5.0             # % - 深度CV良好標準
    depth_cv_acceptable: float = 10.0      # % - 深度CV可接受標準
    
    depth_range_stable: float = 500.0      # mm - 穩定範圍
    depth_range_moderate: float = 1000.0   # mm - 中等範圍
    
    depth_spike_ratio: float = 0.05        # 深度跳動比例閾值
    depth_spike_min_mm: float = 30.0       # mm - 最小深度跳動
    
    depth_outlier_sigma: float = 3.0       # Z-score 異常值閾值
    
    ankle_diff_excellent: float = 20.0     # mm - 腳踝深度差異優秀標準
    ankle_diff_acceptable: float = 50.0    # mm - 腳踝深度差異可接受標準
    
    wrist_depth_tolerance: float = 150.0   # mm - 手腕深度容差
    knee_depth_tolerance: float = 150.0    # mm - 膝蓋深度容差
    
    proximity_radius: float = 500.0        # mm - 接近檢測半徑
    depth_margin: float = 50.0             # mm - 深度容差
    
    # ========== Step 4: 物理運動邏輯 ==========
    max_wrist_speed_ms: float = 15.0       # m/s - 手腕最大速度
    max_ball_speed_ms: float = 30.0        # m/s - 網球最大速度
    
    joint_angle_min: float = 5.0           # 度 - 關節最小角度
    joint_angle_max: float = 175.0         # 度 - 關節最大角度
    
    torso_stability_threshold: float = 50.0  # mm - 軀幹穩定性閾值
    
    racket_contact_threshold: float = 200.0  # mm - 球拍接觸判定距離
    
    max_inter_frame_distance: float = 150.0 # mm - 連續幀之間的最大合理位移

    gravity_acceleration: float = -9810.0  # mm/s² - 重力加速度（負值表示向下）
    gravity_axis: int = 1                  # 重力軸索引 (0=X, 1=Y, 2=Z)
    gravity_tolerance: float = 0.5         # 重力偏差容許比例
    
    energy_drift_threshold: float = 0.3    # 能量漂移閾值
    
    continuity_jump_sigma: float = 3.0     # 運動連續性跳動閾值
    continuity_max_jumps: int = 3          # 最大允許跳動次數
    
    # ========== Step 5: 空間一致性 ==========
    ground_plane_tolerance: float = 30.0   # mm - 地面平面容差
    
    body_symmetry_excellent: float = 2.0   # 身體對稱性優秀標準 (百分比)
    body_symmetry_acceptable: float = 5.0  # 身體對稱性可接受標準
    
    center_stability_sigma: float = 3.0    # 重心穩定性異常閾值
    center_stability_axes: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    
    distance_cv_excellent: float = 2.0     # % - 距離CV優秀標準
    distance_cv_acceptable: float = 5.0    # % - 距離CV可接受標準
    
    penetration_tolerance: float = 30.0    # mm - 穿透容差
    
    rigid_group_cv_threshold: float = 5.0  # % - 剛體組CV閾值
    
    # ========== Step 6: 時間平滑性 ==========
    fps: int = 30                          # 預設幀率
    
    savgol_window: int = 15                # Savitzky-Golay 濾波器窗口
    savgol_polyorder: int = 3              # Savitzky-Golay 多項式階數
    
    jerk_peak_sigma: float = 3.0           # Jerk 峰值檢測閾值
    
    speed_static_threshold: float = 0.05   # m/s - 靜止速度閾值
    speed_moving_threshold: float = 0.15   # m/s - 移動速度閾值
    
    acceleration_static: float = 0.5       # m/s² - 靜止加速度閾值
    
    high_freq_noise_ratio: float = 0.40    # 高頻噪聲比例閾值
    high_frequency_threshold: float = 5.0  # Hz - 判定高頻能量的臨界值
    fft_analysis_joint: str = "right_wrist"  # FFT 分析的關節名稱
    
    direction_continuity_threshold: float = 0.5  # 方向連續性閾值 (cos值)
    direction_change_sudden: float = 75.0        # 度 - 方向突變判斷閾值
    direction_reversal_threshold: float = 150.0  # 度 - 方向反轉判斷閾值
    
    hampel_window: int = 11                # Hampel 濾波器窗口
    hampel_n_sigma: float = 3.0            # Hampel 異常值閾值

    acceleration_sigma: float = 3.0        # 加速度異常檢測的 Z-score 閾值
    jump_detection_sigma: float = 3.0      # 幀間位移異常的 Z-score 閾值
    max_frame_displacement: float = 250.0  # mm - 單幀最大合理位移

    pca_stability_window: int = 21         # PCA 穩定性分析窗口
    
    # ========== 通用參數 ==========
    epsilon: float = 1e-6                  # 數值精度
    
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ValidationConfig':
        """
        從 JSON 檔案載入配置
        
        參數:
            json_path: JSON 配置檔案路徑
        
        返回:
            ValidationConfig: 配置實例
        
        範例:
            >>> config = ValidationConfig.from_json('config.json')
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    
    def to_json(self, json_path: str, indent: int = 2) -> str:
        """
        將配置保存為 JSON 檔案
        
        參數:
            json_path: 輸出檔案路徑
            indent: JSON 縮排空格數
        
        返回:
            str: 輸出檔案路徑
        
        範例:
            >>> config = ValidationConfig()
            >>> config.to_json('my_config.json')
        """
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=indent, ensure_ascii=False)
        return json_path
    
    
    def get_quality_level_cv(self, cv_value: float) -> str:
        """
        根據 CV 值返回品質等級
        
        參數:
            cv_value: 變異係數值（百分比）
        
        返回:
            str: 品質等級 ('優秀', '良好', '可接受', '需改進')
        """
        if cv_value < self.bone_cv_excellent:
            return '優秀'
        elif cv_value < self.bone_cv_good:
            return '良好'
        elif cv_value < self.bone_cv_acceptable:
            return '可接受'
        else:
            return '需改進'
    
    
    def get_quality_level_reprojection(self, error: float) -> str:
        """
        根據重投影誤差返回品質等級
        
        參數:
            error: 重投影誤差（像素）
        
        返回:
            str: 品質等級
        """
        if error < self.reprojection_excellent:
            return '優秀'
        elif error < self.reprojection_good:
            return '良好'
        elif error < self.reprojection_acceptable:
            return '可接受'
        else:
            return '需改進'
    
    
    def get_symmetry_assessment(self, diff_rate: float) -> str:
        """
        根據對稱性差異率返回評估結果
        
        參數:
            diff_rate: 差異率（百分比）
        
        返回:
            str: 評估結果
        """
        if diff_rate < self.bone_symmetry_excellent:
            return '[OK] 對稱'
        elif diff_rate < self.bone_symmetry_acceptable:
            return '[!] 可接受'
        else:
            return '[X] 不對稱'
    
    
    def __repr__(self) -> str:
        """返回配置的字串表示"""
        return f"ValidationConfig(cv_excellent={self.bone_cv_excellent}%, ...)"


# ========== 預設配置實例 ==========
DEFAULT_CONFIG = ValidationConfig()


# ========== 配置載入工具函數 ==========

def load_config(config_path: Optional[str] = None) -> ValidationConfig:
    """
    載入驗證配置
    
    參數:
        config_path: 配置檔案路徑，若為 None 則使用預設配置
    
    返回:
        ValidationConfig: 配置實例
    
    範例:
        >>> config = load_config('my_config.json')
        >>> config = load_config()  # 使用預設配置
    """
    if config_path is None:
        return DEFAULT_CONFIG
    
    if not Path(config_path).exists():
        print(f"警告: 配置檔案 {config_path} 不存在，使用預設配置")
        return DEFAULT_CONFIG
    
    try:
        return ValidationConfig.from_json(config_path)
    except Exception as e:
        print(f"警告: 載入配置失敗 ({e})，使用預設配置")
        return DEFAULT_CONFIG


def save_default_config(output_path: str = 'validation_config.json') -> str:
    """
    保存預設配置為 JSON 檔案
    
    參數:
        output_path: 輸出檔案路徑
    
    返回:
        str: 輸出檔案路徑
    
    範例:
        >>> save_default_config('my_config.json')
    """
    return DEFAULT_CONFIG.to_json(output_path)


# ========== 主程式 ==========
if __name__ == "__main__":
    # 生成預設配置檔案
    output_path = save_default_config()
    print(f"[OK] 已生成預設配置檔案: {output_path}")
    
    # 顯示部分配置內容
    print(f"\n配置示例:")
    print(f"  骨骼 CV 優秀標準: {DEFAULT_CONFIG.bone_cv_excellent}%")
    print(f"  深度範圍: {DEFAULT_CONFIG.depth_min_mm} - {DEFAULT_CONFIG.depth_max_mm} mm")
    print(f"  預設 FPS: {DEFAULT_CONFIG.default_fps}")
