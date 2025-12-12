"""
雙目3D重建驗證 - 共用工具模組
提供所有驗證步驟共用的工具函數
"""

import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


# ============================================================
# 關鍵點提取工具
# ============================================================

def get_keypoint_safely(
    frame: Dict[str, Any],
    keypoint_name: str,
    default: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    安全地從幀中提取關鍵點的 3D 座標
    
    參數:
        frame: 幀數據字典
        keypoint_name: 關鍵點名稱
        default: 無效時返回的預設值
    
    返回:
        np.ndarray: [x, y, z] 座標，若無效則返回 default
    
    範例:
        >>> point = get_keypoint_safely(frame, 'nose')
        >>> if point is not None:
        ...     print(f"鼻子座標: {point}")
    """
    try:
        kp = frame.get(keypoint_name)
        if not isinstance(kp, dict):
            return default
        
        # 檢查是否包含必要的座標
        if all(kp.get(coord) is not None for coord in ['x', 'y', 'z']):
            return np.array([kp['x'], kp['y'], kp['z']], dtype=float)
    except (KeyError, TypeError, ValueError):
        pass
    
    return default


def is_valid_keypoint(frame: Dict[str, Any], keypoint_name: str) -> bool:
    """
    檢查關鍵點是否有效
    
    參數:
        frame: 幀數據字典
        keypoint_name: 關鍵點名稱
    
    返回:
        bool: 關鍵點是否有效
    """
    return get_keypoint_safely(frame, keypoint_name) is not None


def get_keypoint_2d(
    frame: Dict[str, Any],
    keypoint_name: str,
    default: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    安全地提取 2D 關鍵點座標 (x, y)
    
    參數:
        frame: 幀數據字典
        keypoint_name: 關鍵點名稱
        default: 無效時返回的預設值
    
    返回:
        np.ndarray: [x, y] 座標，若無效則返回 default
    """
    try:
        kp = frame.get(keypoint_name)
        if not isinstance(kp, dict):
            return default
        
        if all(kp.get(coord) is not None for coord in ['x', 'y']):
            return np.array([kp['x'], kp['y']], dtype=float)
    except (KeyError, TypeError, ValueError):
        pass
    
    return default


# ============================================================
# 距離與角度計算
# ============================================================

def calculate_distance(
    point1: Optional[np.ndarray],
    point2: Optional[np.ndarray]
) -> Optional[float]:
    """
    計算兩個 3D 點之間的歐式距離
    
    參數:
        point1: 第一個點 [x, y, z]
        point2: 第二個點 [x, y, z]
    
    返回:
        float: 距離值，若任一點無效則返回 None
    """
    if point1 is None or point2 is None:
        return None
    
    try:
        return float(np.linalg.norm(point2 - point1))
    except (ValueError, TypeError):
        return None


def calculate_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray
) -> Optional[float]:
    """
    計算三個點形成的角度（以 B 為頂點）
    
    參數:
        point_a: 第一個點
        point_b: 頂點
        point_c: 第三個點
    
    返回:
        float: 角度（度數），範圍 [0, 180]
    """
    if any(p is None for p in [point_a, point_b, point_c]):
        return None
    
    try:
        v1 = point_a - point_b
        v2 = point_c - point_b
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return None
        
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_theta)
        return float(np.degrees(angle_rad))
    except (ValueError, TypeError):
        return None


def calculate_unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    """
    計算單位向量
    
    參數:
        vector: 輸入向量
    
    返回:
        np.ndarray: 單位向量，若輸入無效則返回 None
    """
    try:
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return None
        return vector / norm
    except (ValueError, TypeError):
        return None


# ============================================================
# 數據載入與驗證
# ============================================================

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    載入並驗證 JSON 檔案
    
    參數:
        file_path: JSON 檔案路徑
    
    返回:
        list: JSON 數據
    
    異常:
        FileNotFoundError: 檔案不存在
        ValueError: JSON 格式錯誤或數據無效
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析錯誤: {e}")
    
    # 驗證數據格式
    if not isinstance(data, list):
        raise ValueError("數據必須是 list 格式")
    
    if len(data) == 0:
        raise ValueError("數據不能為空")
    
    return data


def validate_frame_structure(
    frame: Dict[str, Any],
    required_keypoints: Optional[List[str]] = None
) -> bool:
    """
    驗證幀的數據結構
    
    參數:
        frame: 幀數據
        required_keypoints: 必須存在的關鍵點列表
    
    返回:
        bool: 結構是否有效
    
    異常:
        ValueError: 結構無效時拋出異常
    """
    if not isinstance(frame, dict):
        raise ValueError("幀數據必須是字典格式")
    
    if required_keypoints:
        missing_keypoints = [kp for kp in required_keypoints if kp not in frame]
        if missing_keypoints:
            raise ValueError(f"缺少必要關鍵點: {missing_keypoints}")
    
    # 檢查關鍵點數據格式
    for kp_name, kp_data in frame.items():
        if kp_data is None:
            continue
        
        if not isinstance(kp_data, dict):
            raise ValueError(f"{kp_name} 數據格式錯誤，應為 dict")
        
        # 檢查是否包含座標
        if not all(coord in kp_data for coord in ['x', 'y', 'z']):
            raise ValueError(f"{kp_name} 缺少 x/y/z 座標")
    
    return True


# ============================================================
# JSON 序列化工具
# ============================================================

def convert_to_serializable(obj: Any) -> Any:
    """
    將 NumPy 類型轉換為 Python 原生類型，以便 JSON 序列化
    
    參數:
        obj: 待轉換的對象
    
    返回:
        可序列化的對象
    """
    if isinstance(obj, np.ndarray):
        # 遞歸處理數組中的元素，確保所有元素都被正確轉換
        return [convert_to_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
        val = float(obj)
        # 處理 NaN 和 Infinity，轉換為 None (JSON null)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_json_results(
    results: Dict[str, Any],
    output_path: str,
    ensure_ascii: bool = False,
    indent: int = 2
) -> str:
    """
    保存驗證結果為 JSON 檔案
    
    參數:
        results: 結果字典
        output_path: 輸出路徑
        ensure_ascii: 是否強制 ASCII 編碼
        indent: 縮排空格數
    
    返回:
        str: 輸出檔案路徑
    """
    # 轉換為可序列化格式
    serializable_results = convert_to_serializable(results)
    
    # 保存檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=ensure_ascii, indent=indent)
    
    return output_path


# ============================================================
# 統計分析工具
# ============================================================

def calculate_cv(values: np.ndarray) -> float:
    """
    計算變異係數 (Coefficient of Variation)
    
    參數:
        values: 數值陣列
    
    返回:
        float: CV 值（百分比）
    """
    mean_val = np.mean(values)
    if abs(mean_val) < 1e-6:
        return 0.0
    std_val = np.std(values)
    return float(std_val / abs(mean_val) * 100)


def detect_outliers_iqr(
    values: np.ndarray,
    multiplier: float = 1.5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    使用 IQR 方法檢測異常值
    
    參數:
        values: 數值陣列
        multiplier: IQR 乘數（預設 1.5）
    
    返回:
        tuple: (異常值索引, 統計資訊字典)
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    stats = {
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outlier_count': int(len(outlier_indices)),
        'outlier_rate': float(len(outlier_indices) / len(values) * 100)
    }
    
    return outlier_indices, stats


def detect_outliers_zscore(
    values: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    使用 Z-score 方法檢測異常值
    
    參數:
        values: 數值陣列
        threshold: Z-score 閾值（預設 3.0）
    
    返回:
        tuple: (異常值索引, 統計資訊字典)
    """
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val < 1e-6:
        return np.array([]), {'mean': float(mean_val), 'std': 0.0, 'outlier_count': 0}
    
    z_scores = np.abs((values - mean_val) / std_val)
    outlier_mask = z_scores > threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    stats = {
        'mean': float(mean_val),
        'std': float(std_val),
        'threshold': float(threshold),
        'outlier_count': int(len(outlier_indices)),
        'outlier_rate': float(len(outlier_indices) / len(values) * 100)
    }
    
    return outlier_indices, stats


# ============================================================
# 路徑處理工具
# ============================================================

def generate_output_path(
    input_path: str,
    suffix: str,
    extension: str = '.json'
) -> str:
    """
    生成輸出檔案路徑
    
    參數:
        input_path: 輸入檔案路徑
        suffix: 輸出檔案後綴
        extension: 副檔名
    
    返回:
        str: 輸出檔案路徑
    
    範例:
        >>> generate_output_path('data.json', '_step1_results')
        'data_step1_results.json'
    """
    path = Path(input_path)
    stem = path.stem
    parent = path.parent
    
    output_name = f"{stem}{suffix}{extension}"
    return str(parent / output_name)


# ============================================================
# 關鍵點名稱映射
# ============================================================

KEYPOINT_NAMES_EN = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'tennis_ball'
]

ALL_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'neck', 'left_hand', 'right_hand', 'tennis_ball', 'tennis_racket'
]

KEYPOINT_NAMES_ZH = {
    'nose': '鼻子',
    'left_eye': '左眼',
    'right_eye': '右眼',
    'left_ear': '左耳',
    'right_ear': '右耳',
    'left_shoulder': '左肩',
    'right_shoulder': '右肩',
    'left_elbow': '左肘',
    'right_elbow': '右肘',
    'left_wrist': '左腕',
    'right_wrist': '右腕',
    'left_hip': '左髖',
    'right_hip': '右髖',
    'left_knee': '左膝',
    'right_knee': '右膝',
    'left_ankle': '左踝',
    'right_ankle': '右踝',
    'tennis_ball': '網球'
}


def get_keypoint_name_zh(keypoint_name: str) -> str:
    """
    獲取關鍵點的中文名稱
    
    參數:
        keypoint_name: 英文關鍵點名稱
    
    返回:
        str: 中文名稱，若不存在則返回原名稱
    """
    return KEYPOINT_NAMES_ZH.get(keypoint_name, keypoint_name)
