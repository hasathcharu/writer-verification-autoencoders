import cv2
import numpy as np

def read_image(image_path, resize=True, size=1600):
    image = cv2.imread(image_path)
    if resize:
        image = cv2.resize(image, (size, size))
    return image

def increase_blue_intensity(input):
    hsv = cv2.cvtColor(input.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([70, 20, 20])
    upper_blue = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3)), iterations=4)
    mask = cv2.bitwise_not(mask)

    hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] * 2.5, 0, 255), hsv[:, :, 2])

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def create_rule_mask(input):
    _, binary = cv2.threshold(cv2.cvtColor(input, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    binary_dialted = cv2.dilate(binary, np.ones((4,4), np.uint8), iterations=1)
    line_kernel = np.ones((2, 50), np.uint8)
    horizontal_lines = cv2.morphologyEx(binary_dialted, cv2.MORPH_OPEN, line_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, np.ones((3,3), np.uint8), iterations=4) 
    height, width = horizontal_lines.shape
    M = np.float32([[1, 0, 0], [0, 1, -5]]) 
    horizontal_lines = cv2.warpAffine(horizontal_lines, M, (width, height))
    return horizontal_lines
    
def increase_contrast(input):
    high_constrast_matrix = np.ones(input.shape, dtype='uint8') * 1.1
    return np.uint8(np.clip(cv2.multiply(np.float64(input), high_constrast_matrix), 0, 255))

def remove_rules(input):
    image = input.copy()
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, saturation, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    horizontal_lines = create_rule_mask(image)

    modified = increase_blue_intensity(image)

    _, m_saturation, _ = cv2.split(cv2.cvtColor(modified, cv2.COLOR_BGR2HSV))

    for i in range (modified.shape[0]):
        for j in range (modified.shape[1]):
            if (m_saturation[i][j] < 30 and horizontal_lines[i][j] == 255) or (saturation[i][j] < 15):
                modified[i][j] = 255

    raw_image = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return raw_image, binary

def binarize_only(input):
    image = input.copy()
    image = cv2.GaussianBlur(image, (5, 5), 0)

    raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return raw_image, binary
