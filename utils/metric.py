import numpy as np
import os 
import cv2
from pie import get_keypoint, get_keypoints, get_triangle_flag, get_bboxs, get_angle, get_percent
from ground_truth import extract_gt_bboxes, calculate_triangle_angle, calculate_triangle_angle, calculate_region_percentage
def compute_score(x, y):
    """Tính toán score(i, j) dựa trên công thức đã cho."""
    m, n = len(x), len(y)
    score = np.zeros((m + 1, n + 1))  # score matrix

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # compute the math score
            match_score = 1 - abs(x[i - 1] - y[j - 1]) / y[j - 1]  
            score[i, j] = max(score[i - 1, j],
                              score[i, j - 1], 
                              score[i - 1, j - 1] + match_score)

    return score[-1, -1]/m
def load_masks_from_yolo(image_path):
    """
    Hàm giả lập để tải mask từ mô hình YOLO.
    Cần thay thế bằng code trích xuất mask từ mô hình YOLO thực tế.

    Args:
        image_path (str): Đường dẫn ảnh test.

    Returns:
        np.ndarray: Danh sách các masks dưới dạng numpy array.
    """
    # TODO: Thay thế phần này bằng việc lấy output từ YOLO
    # Giả lập dữ liệu mask: Mảng nhị phân ngẫu nhiên có kích thước (H, W)
    dummy_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    return [dummy_mask]  # Danh sách các masks

def evaluate_segmentation(test_images_dir, json_gt_path):
    """
    Đánh giá mô hình phân đoạn biểu đồ tròn trên toàn bộ tập test.
    
    Args:
        test_images_dir (str): Thư mục chứa các ảnh test.
        json_gt_path (str): Đường dẫn đến file JSON chứa ground truth.

    Returns:
        dict: Kết quả đánh giá trung bình trên toàn bộ tập test.
    """
    all_scores = []
    image_scores = {}
    
    for image_file in os.listdir(test_images_dir):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Lấy ground truth từ file JSON
        gt_bboxes = extract_gt_bboxes(json_gt_path, image_file)
        gt_percentages = np.array(calculate_region_percentage(np.array(gt_bboxes)))
        
        # Lấy giá trị dự đoán từ YOLO segmentation
        image_path = os.path.join(test_images_dir, image_file)
        pred_bboxes = get_bboxs(image_path)

        # Kiểm tra nếu không đủ số điểm
        if len(pred_bboxes) < 3:
            print(f"Warning: Image {image_file} has only {len(pred_bboxes)} detected regions. Skipping...")
            continue  # Bỏ qua ảnh này và tiếp tục với ảnh khác
        
        pred_percentages = np.array(get_percent(pred_bboxes))
        
        # Tính điểm khớp giữa phần trăm dự đoán và ground truth
        score = compute_score(pred_percentages, gt_percentages)
        all_scores.append(score)
        image_scores[image_file] = score
    
    # Trả về kết quả đánh giá trung bình và điểm của từng ảnh
    avg_score = np.mean(all_scores) if all_scores else 0
    return {"average_score": avg_score, "num_images_evaluated": len(all_scores), "image_scores": image_scores}

# Ví dụ sử dụng
if __name__ == "__main__":
    test_images_dir = "ChartInsight/assets/dataset/test/piedata(1008)/pie/images/test2019"
    json_gt_path = "ChartInsight/assets/dataset/test/piedata(1008)/pie/annotations/instancesPie(1008)_test2019.json"
    
    results = evaluate_segmentation(test_images_dir, json_gt_path)
    print("Evaluation Results:", results)
