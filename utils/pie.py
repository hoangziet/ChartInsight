import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform, cdist

def get_keypoint(mask: np.ndarray) -> np.ndarray:
    """
    Trích xuất 3 keypoints từ mask bằng cách tìm 2 điểm xa nhất và 1 điểm xa nhất đến đường nối 2 điểm đó.

    Args:
        mask (np.ndarray): Mask từ kết quả phân đoạn của YOLO.

    Returns:
        np.ndarray: Mảng (3,2) chứa 3 keypoints, hoặc None nếu không có đủ điểm hợp lệ.
    """
    if mask is None or not isinstance(mask, np.ndarray):
        raise ValueError("Mask đầu vào phải là một mảng numpy hợp lệ.")

    # Chuyển mask sang binary (0 hoặc 255)
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # Tìm contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("Không tìm thấy contours trong mask.")

    # Chọn contour có diện tích lớn nhất
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()

    if contour.ndim != 2 or contour.shape[0] < 3:
        raise ValueError("Contour không hợp lệ hoặc có ít hơn 3 điểm.")

    # Tính ma trận khoảng cách giữa các điểm trong contour
    dist_matrix = squareform(pdist(contour))

    # Lấy hai điểm xa nhất
    idx1, idx2 = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)

    # Tìm điểm xa nhất với đường nối hai điểm trên
    idx3 = np.argmax(np.min(dist_matrix[[idx1, idx2]], axis=0))

    keypoints = contour[[idx1, idx2, idx3]]

    return keypoints

def get_keypoints(masks: np.ndarray) -> np.ndarray:
    """
    Get keypoints of triangle from masks, each keypoint box format [x1, y1, x2, y2, xc, yc] (can be shuflled but keep the x, y order)
    Args:
        masks (np.ndarray): masks result from YOLO segmentation

    Returns:
        np.ndarray: shape (n, 6) with n is the number of mask segmented
    """
    keypoints = np.array([get_keypoint(mask) for mask in masks])

    return keypoints

def get_triangle_flag(keypoints: np.ndarray) -> np.ndarray:
    """
    Get the boolean flag of the points in the triangle whether it is the center or not.
    
    Args:
        keypoints (np.ndarray): Keypoints of the triangle, shape (n, 2).
    
    Returns:
        np.ndarray: Shape (n, 3), where n is the number of keypoints.
    """
    if keypoints.shape[0] < 3:
        raise ValueError(f"Error: keypoints must have at least 3 points, but got {keypoints.shape[0]}.")

    t1, t2, t3 = keypoints[:3]  # Lấy 3 điểm đầu tiên

    # Đảm bảo dữ liệu là 2D
    t1, t2, t3 = np.atleast_2d(t1), np.atleast_2d(t2), np.atleast_2d(t3)

    # Tính khoảng cách giữa các điểm của t1 và t2
    dis_12 = cdist(t1, t2, metric="euclidean").ravel()

    # Lấy chỉ số của hai khoảng cách nhỏ nhất
    sorted_indices = np.argsort(dis_12)[:2]
    idx1 = np.unravel_index(sorted_indices[0], (t1.shape[0], t2.shape[0]))
    idx2 = np.unravel_index(sorted_indices[1], (t1.shape[0], t2.shape[0]))

    # Lấy hai cặp điểm gần nhau nhất
    pairs1 = t1[idx1[0]], t2[idx1[1]]
    pairs2 = t1[idx2[0]], t2[idx2[1]]

    # Tính khoảng cách từ t3 đến các cặp điểm gần nhau
    pairs_combined = np.vstack([pairs1, pairs2])
    dis_3 = cdist(t3, pairs_combined, metric="euclidean")

    # Lấy điểm gần nhất với t3 làm "center"
    idx = np.unravel_index(dis_3.argmin(), dis_3.shape)
    point = pairs_combined[idx[1]]  # Lấy giá trị tọa độ thay vì chỉ số

    # Tạo mảng flag với giá trị mặc định là 0
    flag = np.zeros((keypoints.shape[0], 3))

    for i in range(keypoints.shape[0]):
        t = np.atleast_2d(keypoints[i])
        dis = cdist(t, point.reshape(1, -1), metric="euclidean")
        idx = np.unravel_index(dis.argmin(), dis.shape)
        flag[i, idx[0]] = 1

    return flag

def get_bboxs(masks: np.ndarray) -> np.ndarray:
    """
    Get the bounding box of the masks, each bbox format [x1, y1, x2, y2, xc, yc]
    Args:
        masks (np.ndarray): masks result from YOLO segmentation

    Returns:
        np.ndarray: shape (n, 3, 2) with n is the number of mask segmented
    """
    keypoints = get_keypoints(masks)
    flag = get_triangle_flag(keypoints)

    sorted_triangles = np.array(
        [tri[np.argsort(f)] for tri, f in zip(keypoints, flag)]  # flag 1 go end
    )

    return sorted_triangles

def get_angle(bbox: np.ndarray) -> np.ndarray:
    """
    Get the angle BAC of the triangle bounding box in degree
    Args:
        bbox (np.ndarray): bounding box of the mask, format [xA, yA, xB, yB, xc, yc]

    Returns:
        np.ndarray: angle of the bounding box
    """

    vector_CA = bbox[2] - bbox[0]
    vector_CB = bbox[2] - bbox[1]
    
    dot_product = np.dot(vector_CA, vector_CB)
    
    norm_CA = np.linalg.norm(vector_CA)
    norm_CB = np.linalg.norm(vector_CB)
    
    cos_theta = dot_product / (norm_CA * norm_CB)
    angle = np.arccos(cos_theta)
    
    return np.degrees(angle)

def get_percent(bbox: np.ndarray) -> np.ndarray:
    """
    Get the percent of the arc in the pie chart.
    Args:
        bbox (np.ndarray): bounding box of the mask, format [xA, yA, xB, yB, xc, yc]

    Returns:
        np.ndarray: percent of the bounding box
    """
    angles = np.array([get_angle(b) for b in bbox])
    percent = angles / angles.sum()
    return percent



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