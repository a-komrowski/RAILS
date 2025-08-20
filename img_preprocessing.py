import cv2
import numpy as np
from PIL import Image, ImageFilter

class ImagePreprocessor:
    """Collection of image preprocessing methods for railway images."""
    
    @staticmethod
    def method_1_histogram_equalization(img):
        """
        Method 1: Histogram Equalization
        Enhances contrast by redistributing pixel intensities.
        Good for: Dark images, low contrast situations.
        """
        if len(img.shape) == 3:
            # Apply to each channel separately
            img_eq = np.zeros_like(img)
            for i in range(3):
                img_eq[:, :, i] = cv2.equalizeHist(img[:, :, i])
        else:
            img_eq = cv2.equalizeHist(img)
        return img_eq
    
    @staticmethod
    def method_2_clahe(img):
        """
        Method 2: Contrast Limited Adaptive Histogram Equalization (CLAHE)
        Adaptive histogram equalization with contrast limiting.
        Good for: Uneven lighting, avoiding over-amplification of noise.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        if len(img.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_clahe = clahe.apply(img)
        return img_clahe
    
    @staticmethod
    def method_3_gamma_correction(img, gamma=0.7):
        """
        Method 3: Gamma Correction
        Non-linear intensity transformation to brighten dark regions.
        Good for: Dark images, improving visibility of track details.
        """
        # Normalize to [0, 1]
        img_norm = img.astype(np.float32) / 255.0
        # Apply gamma correction
        img_gamma = np.power(img_norm, gamma)
        # Convert back to [0, 255]
        img_gamma = (img_gamma * 255).astype(np.uint8)
        return img_gamma
    
    @staticmethod
    def method_4_unsharp_masking(img):
        """
        Method 4: Unsharp Masking
        Sharpens image by subtracting blurred version.
        Good for: Enhancing edge details, improving track/ballast distinction.
        """
        if len(img.shape) == 3:
            # Convert to PIL for easier processing
            pil_img = Image.fromarray(img)
            # Apply unsharp mask filter
            enhanced = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            img_sharp = np.array(enhanced)
        else:
            # For grayscale, use Gaussian blur and subtract
            blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
            img_sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return img_sharp
    
    @staticmethod
    def method_5_edge_enhancement(img):
        """
        Method 5: Edge Enhancement + Morphological Operations
        Combines edge detection with morphological operations.
        Good for: Highlighting infrastructure boundaries, track edges.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and convert to uint8
        sobel_normalized = ((sobel_combined / sobel_combined.max()) * 255).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_cleaned = cv2.morphologyEx(sobel_normalized, cv2.MORPH_CLOSE, kernel)
        
        # Combine with original
        if len(img.shape) == 3:
            # Create 3-channel edge image
            edges_3ch = cv2.cvtColor(edges_cleaned, cv2.COLOR_GRAY2RGB)
            img_enhanced = cv2.addWeighted(img, 0.7, edges_3ch, 0.3, 0)
        else:
            img_enhanced = cv2.addWeighted(gray, 0.7, edges_cleaned, 0.3, 0)
        
        return img_enhanced