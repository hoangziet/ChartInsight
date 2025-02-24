import numpy as np
import cv2
from scipy.spatial.distance import pdist, squareform, cdist

def get_keypoint(mask: np.ndarray) -> np.ndarray : 
    """
    Get 3 keypoint from mask by finding the 2 farthest points and the farthest point to the closest point between the 2 farthest points

    Args:
        mask (np.ndarray): mask in masks result from YOLO segmentation

    Returns:
        np.ndarray: 3 keypoints
    """
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze()
    
    dist_matrix = squareform(pdist(contour))
    
    idx1, idx2 = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    idx3 = np.argmax(np.min(dist_matrix[[idx1, idx2]], axis = 0))
    
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
    Get the boolean flag of the points in the triangle whether it is the center or not 
    Args:
        keypoints (np.ndarray): keypoints of the triangle
    Returns:
        np.ndarray: shape (n, 3) with n is the number of keypoints, 3 is the number of points in the triangle
    """    
    if keypoints.shape[0] >= 3: 
        t1, t2, t3 = keypoints[:3]
        
        # find the 2 closest pairs of t1 and t2
        dis_12 = cdist(t1, t2, metric = "euclidean")
        
        _ = dis_12.ravel()

        # get the position of the first and second smallest
        pos1, pos2 = np.argsort(_)[:2]

        # unravel it to get the index of the points
        idx1 = np.unravel_index(pos1, dis_12.shape)
        idx2 = np.unravel_index(pos2, dis_12.shape)
        
        # get the pairs
        idx1_x, idx1_y = idx1
        pairs1 = t1[idx1_x], t2[idx1_y]

        idx2_x, idx2_y = idx2
        pairs2 = t1[idx2_x], t2[idx2_y]        
        
        # compare with the third triangle to find the closest point
        dis_3 = cdist(t3, np.vstack([pairs1, pairs2]), metric = "euclidean")
        idx = np.unravel_index(dis_3.argmin(), dis_3.shape)
        
        # IT IS THE CENTER!!!!
        point = t3[idx[0]]
        
    flag = np.zeros((keypoints.shape[0], 3))
    for i in range(keypoints.shape[0]):
        t = keypoints[i]
        dis = cdist(t, point.reshape(1, -1), metric = "euclidean")
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