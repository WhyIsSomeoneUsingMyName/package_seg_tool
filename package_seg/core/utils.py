import cv2
import numpy as np

def keep_largest_component(binary_mask):
    """保留面积最大的连通区域（基于轮廓）"""
    if binary_mask is None or binary_mask.sum() == 0:
        return binary_mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_mask
    largest_contour = max(contours, key=cv2.contourArea)
    result = np.zeros_like(binary_mask)
    cv2.fillPoly(result, [largest_contour], 255)
    return result