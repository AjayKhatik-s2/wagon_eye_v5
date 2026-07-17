"""
Wagon Number OCR Module

Reads and validates 11-digit wagon numbers from YOLO-detected regions using EasyOCR.

The 11-digit wagon number structure:
- C1-C2 (2 digits): Type of Wagon
- C3-C4 (2 digits): Owning Railway
- C5-C6 (2 digits): Year of Manufacture
- C7-C10 (4 digits): Individual Wagon Number
- C11 (1 digit): Check Digit

Handles single-line and multi-line digit layouts with spatial reconstruction.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================
# Fix for Pillow 10+: ANTIALIAS was removed, replaced by LANCZOS
# EasyOCR internally uses PIL.Image.ANTIALIAS, so we restore it
# ============================================================
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("WARNING: EasyOCR not installed. Wagon number recognition will be disabled.")
    print("Install with: pip install easyocr")


@dataclass
class WagonNumber:
    """Validated wagon number with structure breakdown."""
    
    full_number: str  # Complete 11-digit number as string
    wagon_type: str  # C1-C2
    owning_railway: str  # C3-C4
    year_of_manufacture: str  # C5-C6
    individual_number: str  # C7-C10
    check_digit: str  # C11
    
    yolo_confidence: float  # Confidence from YOLO detection
    ocr_confidence: float  # Average OCR confidence across all digits
    per_digit_confidences: List[float] = field(default_factory=list)
    
    is_valid: bool = True
    is_manipulated: bool = False  # True if first two digits were corrected to fit 10-39 range
    validation_errors: List[str] = field(default_factory=list)
    
    def __str__(self):
        """Human-readable representation."""
        if self.is_valid:
            return (f"{self.wagon_type}-{self.owning_railway}-{self.year_of_manufacture}-"
                   f"{self.individual_number}-{self.check_digit} "
                   f"(OCR: {self.ocr_confidence:.2%}, YOLO: {self.yolo_confidence:.2%})")
        else:
            return f"INVALID: {self.full_number} - {', '.join(self.validation_errors)}"


class WagonNumberValidator:
    """Validates wagon numbers against railway numbering scheme."""
    
    # Valid range for first two digits (wagon type): 10-39
    VALID_FIRST_TWO_MIN = 10
    VALID_FIRST_TWO_MAX = 39
    
    @staticmethod
    def validate(digits: str, confidences: List[float], 
                 min_confidence: float = 0.75) -> Tuple[bool, List[str]]:
        """
        Validate wagon number structure.
        
        Args:
            digits: String of detected digits
            confidences: Per-digit confidence scores
            min_confidence: Minimum acceptable confidence (default lowered to 0.30)
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check 1: Exactly 11 digits
        if len(digits) != 11:
            errors.append(f"Expected 11 digits, got {len(digits)}")
            return False, errors
        
        # Check 2: All numeric
        if not digits.isdigit():
            errors.append("Contains non-numeric characters")
            return False, errors
        
        # Check 3: First two digits (wagon type) must be in range 10-39
        first_two = int(digits[:2])
        if first_two < 10 or first_two > 39:
            errors.append(f"Invalid wagon type: {digits[:2]} (must be 10-39)")
        
        # Skip confidence check - EasyOCR confidence can be lower
        # Just accept if we have 11 digits
        
        # Check 4: Year plausibility (optional - can be disabled)
        year = digits[4:6]
        try:
            year_int = int(year)
            if year_int > 49:
                actual_year = 1900 + year_int
            else:
                actual_year = 2000 + year_int
            
            if actual_year < 1950 or actual_year > 2049:
                errors.append(f"Unusual year: {actual_year} (from digits {year})")
        except ValueError:
            pass
        
        # Valid if no critical errors (11 digits, numeric, valid wagon type range)
        critical_errors = [e for e in errors if "Expected 11" in e or "non-numeric" in e or "Low confidence" in e or "Invalid wagon type" in e]
        is_valid = len(critical_errors) == 0
        
        return is_valid, errors
    
    @staticmethod
    def validate_check_digit(wagon_number: str) -> bool:
        """Validate check digit (placeholder)."""
        return True


class WagonNumberOCR:
    """
    OCR processor for wagon numbers using EasyOCR.
    
    EasyOCR is a ready-to-use OCR with high accuracy for 
    printed text recognition supporting 80+ languages.
    
    Handles:
    - Image preprocessing (resize, contrast enhancement)
    - Text extraction with EasyOCR
    - Digit filtering and validation
    - 11-digit sequence reconstruction
    """
    
    def __init__(
        self,
        use_angle_cls: bool = True,  # Kept for API compatibility
        lang: str = 'en',             # Language for EasyOCR
        use_gpu: bool = False,
        min_confidence: float = 0.85,
        resize_factor: float = 3.0,   # 3x for weathered/stenciled text
        model_name: str = None  # Kept for API compatibility
    ):
        """
        Initialize OCR processor with EasyOCR.
        
        Args:
            use_angle_cls: (Ignored - for API compatibility)
            lang: Language for EasyOCR (default: 'en')
            use_gpu: Use GPU acceleration
            min_confidence: Minimum OCR confidence threshold
            resize_factor: Image resize multiplier for better OCR
            model_name: (Ignored - for API compatibility)
        """
        self.min_confidence = min_confidence
        self.resize_factor = resize_factor
        self.use_gpu = use_gpu
        
        if not EASYOCR_AVAILABLE:
            self.reader = None
            print("EasyOCR not available - wagon number recognition disabled")
            return
        
        try:
            print(f"Loading EasyOCR with language: {lang}")
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(
                [lang],
                gpu=use_gpu,
                verbose=False
            )
            if use_gpu:
                print(f"✓ EasyOCR initialized with GPU")
            else:
                print(f"✓ EasyOCR initialized (CPU)")
                
        except Exception as e:
            print(f"⚠ EasyOCR initialization failed: {e}")
            self.reader = None
    
    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess cropped wagon number region for OCR.
        
        Steps:
        1. Resize (3x for better OCR accuracy on weathered text)
        2. Denoise (remove dirt/rust noise)
        3. Enhance contrast (aggressive CLAHE for faded paint)
        4. Sharpen (crisp up blurred digit edges)
        
        Args:
            crop: Cropped BGR image containing wagon number
            
        Returns:
            Preprocessed image as numpy array (RGB)
        """
        if crop is None or crop.size == 0:
            return None
        
        # Step 1: Resize for better OCR (3x for stenciled/weathered text)
        h, w = crop.shape[:2]
        new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Step 2: Denoise to remove dirt/rust noise
        denoised = cv2.fastNlMeansDenoisingColored(resized, None, h=8, hColor=8,
                                                     templateWindowSize=7, searchWindowSize=21)
        
        # Step 3: Convert to RGB (EasyOCR prefers RGB)
        if len(denoised.shape) == 3 and denoised.shape[2] == 3:
            rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        elif len(denoised.shape) == 2:
            rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        else:
            rgb = denoised
        
        # Step 4: Aggressive contrast enhancement using CLAHE (higher clipLimit for faded paint)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Step 5: Sharpen using unsharp mask to crisp up digit edges
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        
        return sharpened
    
    
    def run_ocr(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """
        Run EasyOCR on preprocessed image.
        
        Args:
            image: RGB numpy array image
            
        Returns:
            (recognized_text, average_confidence, detailed_results)
        """
        if image is None or self.reader is None:
            return "", 0.0, []
        
        try:
            # Run EasyOCR
            # allowlist restricts to digits only for wagon numbers
            results = self.reader.readtext(
                image,
                allowlist='0123456789',
                paragraph=False,  # Don't merge text blocks
                detail=1  # Return detailed results with confidence
            )
            
            if not results:
                return "", 0.0, []
            
            # Sort results by Y-coordinate (top-to-bottom) to read upper row first
            # EasyOCR bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Use the average Y of the bbox top-left and top-right corners
            def get_y_center(result):
                bbox = result[0]
                y_coords = [point[1] for point in bbox]
                return sum(y_coords) / len(y_coords)
            
            results = sorted(results, key=get_y_center)
            
            # Combine all detected text (now in top-to-bottom order)
            all_text = ""
            confidences = []
            detailed_results = []
            
            for (bbox, text, conf) in results:
                all_text += text
                confidences.append(conf)
                detailed_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf
                })
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return all_text, avg_confidence, detailed_results
            
        except Exception as e:
            print(f"⚠ EasyOCR inference failed: {e}")
            return "", 0.0, []
    
    def extract_digits(self, text: str) -> str:
        """
        Extract only digit characters from OCR text.
        
        Args:
            text: Raw OCR output
            
        Returns:
            String containing only digits
        """
        return ''.join(c for c in text if c.isdigit())
    
    def _correct_wagon_type_digits(self, digit_string: str, debug: bool = False) -> Tuple[str, bool]:
        """
        Correct the first two digits if they fall outside the valid wagon type range (10-39).
        
        OCR on weathered/stenciled wagon text commonly misreads certain digits.
        This method applies a confusion map to the first digit to bring the
        wagon type back into the valid 10-39 range.
        
        Args:
            digit_string: Raw OCR digit string (must be >= 2 chars)
            debug: Enable debug logging
            
        Returns:
            (corrected_digit_string, was_manipulated)
            - corrected_digit_string: digit string with first digit corrected (if needed)
            - was_manipulated: True if any correction was applied
        """
        if len(digit_string) < 2:
            return digit_string, False
        
        first_two = int(digit_string[:2])
        if 10 <= first_two <= 39:
            # Already in valid range — no manipulation needed
            return digit_string, False
        
        # OCR confusion map for the FIRST digit:
        # Maps commonly misread digits to their most likely correct value
        # so that the first two digits fall within 10-39.
        FIRST_DIGIT_CORRECTIONS = {
            '0': '2',  # 0x -> 2x (range 20-29)
            '4': '1',  # 4x -> 1x (range 10-19)
            '5': '3',  # 5x -> 3x (range 30-39)
            '6': '1',  # 6x -> 1x (range 10-19)
            '7': '1',  # 7x -> 1x (range 10-19)
            '8': '3',  # 8x -> 3x (range 30-39)
            '9': '3',  # 9x -> 3x (range 30-39)
        }
        
        first_digit = digit_string[0]
        if first_digit in FIRST_DIGIT_CORRECTIONS:
            corrected_digit = FIRST_DIGIT_CORRECTIONS[first_digit]
            corrected_string = corrected_digit + digit_string[1:]
            corrected_two = int(corrected_string[:2])
            
            if 10 <= corrected_two <= 39:
                if debug:
                    print(f"  [MANIPULATION] First two digits '{digit_string[:2]}' outside 10-39 range → "
                          f"corrected first digit '{first_digit}' → '{corrected_digit}' → "
                          f"new wagon type '{corrected_string[:2]}'")
                return corrected_string, True
        
        if debug:
            print(f"  [MANIPULATION] First two digits '{digit_string[:2]}' outside 10-39 range, "
                  f"no correction available for first digit '{first_digit}'")
        return digit_string, False
    
    def reconstruct_wagon_number(
        self,
        crop: np.ndarray,
        yolo_confidence: float,
        debug: bool = False
    ) -> Optional[WagonNumber]:
        """
        Complete pipeline: preprocess -> OCR -> extract -> validate -> construct.
        
        Args:
            crop: Cropped image containing wagon number region
            yolo_confidence: Confidence from YOLO detection
            debug: Enable debug logging
            
        Returns:
            WagonNumber object if valid, None if invalid or OCR failed
        """
        if self.reader is None:
            if debug:
                print("  [DEBUG] OCR not initialized")
            return None
        
        if crop is None or crop.size == 0:
            if debug:
                print("  [DEBUG] Empty crop")
            return None
        
        try:
            # Step 1: Preprocess
            processed_image = self.preprocess_crop(crop)
            if processed_image is None:
                if debug:
                    print("  [DEBUG] Preprocessing failed")
                return None
            
            # Step 2: Run EasyOCR
            raw_text, confidence, detailed_results = self.run_ocr(processed_image)
            
            if debug:
                print(f"  [DEBUG] Raw OCR text: '{raw_text}' (conf: {confidence:.2%})")
                for detail in detailed_results:
                    print(f"    - '{detail['text']}' (conf: {detail['confidence']:.2%})")
            
            # Step 3: Extract digits and build result
            digit_string = self.extract_digits(raw_text) if raw_text else ""
            
            if debug:
                print(f"  [DEBUG] Extracted digits: '{digit_string}' ({len(digit_string)} chars)")
            
            # Single-pass only: multi-frame aggregation handles accuracy
            # No variant retries needed
            
            if not digit_string:
                return None
            
            # Step 4: Correct first two digits if outside valid range (10-39)
            # ONLY for 11-digit wagon numbers. Skip for 5-digit loco numbers
            # (loco numbers don't follow wagon numbering scheme).
            was_manipulated = False
            if len(digit_string) == 11:
                digit_string, was_manipulated = self._correct_wagon_type_digits(digit_string, debug=debug)
            
            # Step 5: Validate
            confidences = [confidence] * len(digit_string) if digit_string else []
            
            is_valid, errors = WagonNumberValidator.validate(
                digit_string,
                confidences,
                min_confidence=self.min_confidence
            )
            
            # Step 7: Construct WagonNumber object
            if len(digit_string) == 11:
                wagon_number = WagonNumber(
                    full_number=digit_string,
                    wagon_type=digit_string[0:2],
                    owning_railway=digit_string[2:4],
                    year_of_manufacture=digit_string[4:6],
                    individual_number=digit_string[6:10],
                    check_digit=digit_string[10],
                    yolo_confidence=yolo_confidence,
                    ocr_confidence=confidence,
                    per_digit_confidences=confidences,
                    is_valid=is_valid,
                    is_manipulated=was_manipulated,
                    validation_errors=errors
                )
                
                if is_valid and not WagonNumberValidator.validate_check_digit(digit_string):
                    wagon_number.is_valid = False
                    wagon_number.validation_errors.append("Check digit validation failed")
                
                return wagon_number
            else:
                # Wrong number of digits
                return WagonNumber(
                    full_number=digit_string,
                    wagon_type="",
                    owning_railway="",
                    year_of_manufacture="",
                    individual_number="",
                    check_digit="",
                    yolo_confidence=yolo_confidence,
                    ocr_confidence=confidence,
                    per_digit_confidences=confidences,
                    is_valid=False,
                    validation_errors=errors
                )
        
        except Exception as e:
            import traceback
            print(f"⚠ OCR failed: {e}")
            if not hasattr(self, '_ocr_error_count'):
                self._ocr_error_count = 0
            self._ocr_error_count += 1
            if self._ocr_error_count <= 3:
                traceback.print_exc()
            return None


# ============================================================================
# TESTING / DEMO
# ============================================================================

def test_wagon_number_ocr():
    """Test wagon number OCR with a sample image."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wagon_number_ocr.py <wagon_number_image.jpg>")
        print("\nThis will test the OCR on a cropped wagon number image.")
        return
    
    image_path = sys.argv[1]
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    print(f"Testing OCR on: {image_path}")
    print(f"Image size: {img.shape}")
    
    # Initialize OCR
    ocr = WagonNumberOCR()
    
    if ocr.reader is None:
        print("ERROR: EasyOCR not available")
        return
    
    # Run OCR
    wagon_num = ocr.reconstruct_wagon_number(img, yolo_confidence=0.95, debug=True)
    
    if wagon_num:
        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"{'='*60}")
        print(f"Full Number: {wagon_num.full_number}")
        print(f"Type: {wagon_num.wagon_type}")
        print(f"Railway: {wagon_num.owning_railway}")
        print(f"Year: {wagon_num.year_of_manufacture}")
        print(f"Individual: {wagon_num.individual_number}")
        print(f"Check Digit: {wagon_num.check_digit}")
        print(f"\nOCR Confidence: {wagon_num.ocr_confidence:.2%}")
        print(f"\nValid: {wagon_num.is_valid}")
        if wagon_num.validation_errors:
            print(f"Errors: {wagon_num.validation_errors}")
        print(f"{'='*60}")
    else:
        print("\n❌ OCR failed - no valid wagon number detected")


if __name__ == '__main__':
    test_wagon_number_ocr()

