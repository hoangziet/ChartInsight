import os
import re
import numpy as np
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import matplotlib.colors as mcolors
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Union, Tuple

# Define paths and models
IMAGE_DIR = "./../assets/dataset/reduced_data/bardata(1031)/bar/images/test2019"
LABEL_DIR = "./../assets/dataset/reduced_data/bardata(1031)/bar/labels/test2019"
OBJECT_DETECTION_MODEL_DIR = "./../training/object_detection/runs/detect/train/weights/best.pt"
BAR_DETECTION_DIR = "./../training/bar_detection/runs/detect/train/weights/best.pt"

# Load models
od_model = YOLO(OBJECT_DETECTION_MODEL_DIR)
bd_model = YOLO(BAR_DETECTION_DIR)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')


def detect_objects(model: YOLO, image: Union[str, np.ndarray], class_id: int) -> Union[np.ndarray, None]:
    """
    Detect objects in an image using a YOLO model.

    Args:
        model: YOLO model.
        image: Image path or image array.
        class_id: Class ID to filter.

    Returns:
        Bounding box (np.ndarray) of the detected object, or None if not found.
    """
    results = model(image)[0].boxes.data.cpu().numpy()
    filtered = results[results[:, 5] == class_id]
    return filtered[0] if len(filtered) > 0 else None


def get_plot_area(image_path: str) -> Union[np.ndarray, None]:
    """
    Extract the plot (chart) area from an image.

    Args:
        image_path: Path to the image.

    Returns:
        Cropped plot image (RGB) or None if not found.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_box = detect_objects(od_model, image_path, class_id=4)
    if plot_box is not None:
        x1, y1, x2, y2 = map(int, plot_box[:4])
        return image[y1:y2, x1:x2]
    return None


def get_bar_anns(image: np.ndarray) -> np.ndarray:
    """
    Detect bars in the plot area.

    Args:
        image: Cropped plot image.

    Returns:
        Bounding boxes of detected bars as a numpy array.
    """
    return bd_model(image)[0].boxes.data.cpu().numpy()


def get_ocr_results(image: np.ndarray) -> List:
    """
    Extract text using OCR from an image.

    Args:
        image: Image to extract text from.

    Returns:
        OCR results (list).
    """
    results = ocr_model.ocr(image, cls=True)
    if results is None or not results:
        print("[DEBUG] OCR found no results!")
        return []
    return results


def is_numeric(text: str) -> bool:
    """
    Check if a string is a numeric value.

    Args:
        text: String to check.

    Returns:
        True if the string represents a number, otherwise False.
    """
    return bool(re.match(r'^\d+(\.\d+)?$', re.sub(r'[^\d.]', '', text)))


def filter_x_labels(ocr_results: List, bar_anns: Union[np.ndarray, List], 
                    threshold: Union[float, None] = None, x_threshold: float = 10) -> List[Dict]:
    """
    Filter and cluster X-axis labels from OCR results.

    If no threshold is provided, it is automatically calculated as the maximum y-value 
    of the bar_anns plus 50.

    Args:
        ocr_results: OCR results.
        bar_anns: List of bounding boxes for bars (numpy array or list).
        threshold: Y-coordinate filtering threshold (if None, it is automatically calculated).
        x_threshold: Distance threshold along the X-axis for clustering labels.

    Returns:
        List of clustered X-axis labels in dictionary format with the keys:
        'text', 'conf', 'x_mean', 'y_mean'.
    """

    if not ocr_results or not ocr_results[0]: 
        print("No OCR results found for the X-axis!")
        return []

    if threshold is None:
        if bar_anns is None:
            raise ValueError("Either bar_anns or threshold must be provided to filter X-axis labels.")
        bar_list = bar_anns.tolist() if isinstance(bar_anns, np.ndarray) else bar_anns
        max_y = max(box[3] for box in bar_list)
        threshold = max_y

    x_labels = []

    for res in ocr_results[0]:
        bbox, (text, conf) = res
        x_mean = np.mean([pt[0] for pt in bbox])
        y_mean = np.mean([pt[1] for pt in bbox])
        if y_mean > threshold:
            x_labels.append({'text': text, 'conf': conf, 'x_mean': x_mean, 'y_mean': y_mean})
    if not x_labels:
        return []

    x_labels_sorted = sorted(x_labels, key=lambda lbl: lbl['x_mean'])
    clusters = []
    current_cluster = [x_labels_sorted[0]]
    for lbl in x_labels_sorted[1:]:
        current_center = np.mean([l['x_mean'] for l in current_cluster])
        if abs(lbl['x_mean'] - current_center) < x_threshold:
            current_cluster.append(lbl)
        else:
            clusters.append(current_cluster)
            current_cluster = [lbl]
    clusters.append(current_cluster)

    merged = [{
        'text': " ".join(l['text'] for l in cluster),
        'conf': np.mean([l['conf'] for l in cluster]),
        'x_mean': np.mean([l['x_mean'] for l in cluster]),
        'y_mean': np.mean([l['y_mean'] for l in cluster])
    } for cluster in clusters]

    return merged


def filter_y_labels(ocr_results: List, bars_detected: Union[np.ndarray, List]) -> List[Dict]:
    """
    Filter Y-axis labels from OCR results based on the x_mean threshold.
    The threshold is calculated as: 
        threshold = (minimum x of all bounding boxes of bars) - (avg_width * 0.2)
    
    Args:
        ocr_results: OCR results.
        bars_detected: List of bounding boxes for bars (numpy array or list).

    Returns:
        List of Y-axis labels in dictionary format with the keys: 'text', 'conf', 'x_mean', 'y_mean'.
    """

    if not ocr_results or not ocr_results[0]:  # Check if ocr_results is empty or None
        print("No OCR results found for the Y-axis!")
        return []

    # Convert bars_detected to a list if needed
    bars_list = bars_detected.tolist() if isinstance(bars_detected, np.ndarray) else bars_detected

    # Compute the minimum x value and the average width of the bars
    min_x = min(bar[0] for bar in bars_list)
    # avg_width = np.mean([bar[2] - bar[0] for bar in bars_list])
    # threshold = min_x - (avg_width * 0.2)
    threshold = min_x

    return [
        {
            'text': text,
            'conf': conf,
            'x_mean': np.mean([pt[0] for pt in bbox]),
            'y_mean': np.mean([pt[1] for pt in bbox])
        }
        for res in ocr_results[0]
        for bbox, (text, conf) in [res]
        if np.mean([pt[0] for pt in bbox]) < threshold and is_numeric(text)
    ]


def parse_number(text: str) -> float:
    """
    Convert a string to a floating-point number.

    Args:
        text: String to convert.

    Returns:
        Floating-point number.
    """
    try:
        return float(re.sub(r'[^\d.]', '', text))
    except ValueError:
        return 0.0


def closest_color(hex_color: str) -> str:
    """
    Find the closest color in the CSS4_COLORS range of matplotlib for the given hex color.

    Args:
        hex_color: Hex color code, e.g., "#aabbcc".

    Returns:
        Hex color code of the closest match in CSS4_COLORS.
    """
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    css4 = mcolors.CSS4_COLORS
    min_dist = float('inf')
    closest = None
    for color_hex in css4.values():
        candidate = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        dist = np.linalg.norm(np.array(rgb) - np.array(candidate))
        if dist < min_dist:
            min_dist = dist
            closest = color_hex
    return closest


def get_average_color(image: np.ndarray, bbox: List[float]) -> str:
    """
    Compute the average color of a region defined by a bounding box in an image,
    then map the average color to the closest CSS4_COLORS.

    Args:
        image: Original image (RGB).
        bbox: Bounding box in the format [x1, y1, x2, y2].

    Returns:
        Hex color code of the closest match.
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "#000000"
    avg = np.mean(roi, axis=(0, 1))
    raw_hex = "#{:02x}{:02x}{:02x}".format(int(avg[0]), int(avg[1]), int(avg[2]))
    return closest_color(raw_hex)


def get_y_scale(y_labels_sorted: List[Dict]) -> float:
    """
    Calculate the value per pixel ratio based on Y-axis labels.

    This is based on the center position of the topmost (highest value) 
    and bottommost (lowest value) labels.

    Args:
        y_labels_sorted: List of Y-axis labels sorted in ascending order by y_mean.

    Returns:
        Value per pixel (float).
    """
    top_label = y_labels_sorted[0]
    bottom_label = y_labels_sorted[-1]
    mid_top = top_label['y_mean']
    mid_bottom = bottom_label['y_mean']
    pixel_distance = mid_bottom - mid_top
    value_range = parse_number(top_label['text']) - parse_number(bottom_label['text'])
    if pixel_distance == 0:
        return 0.0
    return value_range / pixel_distance


def get_bar_values(plot_area: np.ndarray) -> List[Dict]:
    """
    Extracts information about bars in a chart from the cropped image.

    Args:
        plot_area: The cropped plot area image.

    Returns:
        A list of dictionaries containing: 'bbox', 'label', 'value', and 'color'.
    """

    # Detect bars in the plot image
    bars = get_bar_anns(plot_area)
    if bars is None or len(bars) == 0:
        print("[DEBUG] No bars found!")
        return []

    # Perform OCR on the image
    ocr_results = get_ocr_results(plot_area)

    # Filter X & Y axis labels
    x_labels = filter_x_labels(ocr_results, bars)
    y_labels = filter_y_labels(ocr_results, bars)

    if len(y_labels) < 2:
        print("[DEBUG] Not enough Y-axis labels to calculate values!")
        return []

    # Calculate the value per pixel ratio from the Y-axis labels
    y_labels_sorted = sorted(y_labels, key=lambda lbl: lbl['y_mean'])
    y_scale = get_y_scale(y_labels_sorted)

    # Sort & cluster bars based on their X-center
    bars_sorted = sorted(bars.tolist(), key=lambda b: (b[0] + b[2]) / 2)
    bar_clusters = []
    current_cluster = [bars_sorted[0]]

    for bar in bars_sorted[1:]:
        center_current = np.mean([(b[0] + b[2]) / 2 for b in current_cluster])
        center_bar = (bar[0] + bar[2]) / 2
        if abs(center_bar - center_current) < 10:  # Clustering threshold
            current_cluster.append(bar)
        else:
            bar_clusters.append(current_cluster)
            current_cluster = [bar]
    bar_clusters.append(current_cluster)

    # Assign labels & calculate values for bars
    bar_data = []
    label_counter = 1
    for cluster in bar_clusters:
        cluster_center = np.mean([(bar[0] + bar[2]) / 2 for bar in cluster])
        best_label = min(x_labels, key=lambda lbl: abs(lbl['x_mean'] - cluster_center), default=None)
        
        label_text = best_label['text'] if best_label and abs(best_label['x_mean'] - cluster_center) < 15 else f"label{label_counter}"
        label_counter += 1

        # Get the average color of the bar group
        union_bbox = [
            min(bar[0] for bar in cluster),
            min(bar[1] for bar in cluster),
            max(bar[2] for bar in cluster),
            max(bar[3] for bar in cluster)
        ]
        group_color = get_average_color(plot_area, union_bbox)

        for bar in cluster:
            bar_value = (bar[3] - bar[1]) * y_scale
            bar_data.append({
                'bbox': bar[:4],
                'label': label_text,
                'value': bar_value,
                'color': group_color
            })

    return bar_data