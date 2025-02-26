import json
import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform, cdist

def extract_gt_bboxes(json_path: str, target_image: str) -> list:
    """
    Trích xuất danh sách bounding box (bbox) từ tệp JSON cho một hình ảnh cụ thể.

    Args:
        json_path (str): Đường dẫn đến tệp JSON chứa annotations.
        target_image (str): Tên hình ảnh cần lấy bbox.

    Returns:
        list[np.ndarray]: Danh sách bbox dưới dạng NumPy array.
    """
    # Đọc file JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Tìm image_id tương ứng với target_image
    image_id = next((img["id"] for img in data["images"] if img["file_name"] == target_image), None)

    if image_id is None:
        print(f"Không tìm thấy {target_image} trong JSON.")
        return []

    # Lấy danh sách bbox từ annotations
    gt_bboxes = [np.array(ann["bbox"]).reshape(-1, 2) for ann in data["annotations"] if ann["image_id"] == image_id]

    return gt_bboxes



def calculate_triangle_angle(triangle_bbox: np.ndarray) -> float:
    """
    Tính góc BAC của bounding box dạng tam giác.

    Args:
        triangle_bbox (np.ndarray): Mảng chứa 3 điểm [xA, yA, xB, yB, xC, yC]

    Returns:
        float: Góc tính bằng độ
    """
    # Đảm bảo triangle_bbox có kích thước (3,2)
    triangle_bbox = triangle_bbox.reshape(3, 2)  # Reshape về dạng (3,2) nếu chưa đúng

    vector_CA = triangle_bbox[2] - triangle_bbox[0]
    vector_CB = triangle_bbox[2] - triangle_bbox[1]

    dot_product = np.dot(vector_CA, vector_CB)

    norm_CA = np.linalg.norm(vector_CA)
    norm_CB = np.linalg.norm(vector_CB)

    cos_theta = dot_product / (norm_CA * norm_CB)

    # Đảm bảo giá trị cos nằm trong khoảng hợp lệ [-1, 1] để tránh lỗi arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)  
    return np.degrees(angle_rad)  
def calculate_region_percentage(bboxes: np.ndarray) -> np.ndarray:
    """
    Tính phần trăm của từng vùng trong biểu đồ hình tròn dựa trên góc của chúng.

    Args:
        bboxes (np.ndarray): Danh sách bounding boxes, mỗi box có dạng [xA, yA, xB, yB, xC, yC]

    Returns:
        np.ndarray: Mảng chứa phần trăm của từng vùng trong pie chart (làm tròn 2 số thập phân)
    """
    # Tính góc cho từng bbox
    angles = np.array([calculate_triangle_angle(bbox) for bbox in bboxes if bbox.shape == (3, 2)])

    # Kiểm tra nếu tổng angles = 0 để tránh lỗi chia cho 0
    total_angle = angles.sum()
    if total_angle == 0:
        return np.zeros_like(angles)  # Trả về mảng toàn 0 nếu tổng góc = 0

    percentages = angles / total_angle

    return percentages

