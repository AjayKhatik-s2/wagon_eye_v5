"""
Illumination Processor Module - Normalize lighting conditions for robust door detection.

Implements:
- Adaptive gamma correction based on frame brightness
- CLAHE on luminance channel for local contrast enhancement
- Edge-preserving shadow suppression using bilateral filtering
- Frame-level illumination quality scoring

All operations preserve door edges and texture while reducing
classification errors caused by glare, shadows, and brightness variation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class IlluminationResult:
    """Result of illumination processing for a single frame."""
    normalized_frame: np.ndarray   # The corrected frame (BGR)
    quality_score: float           # 0.0-1.0 quality score
    is_anomaly: bool               # Flagged for offline review
    anomaly_reason: Optional[str]  # Reason if anomaly
    
    # Individual component scores (for debugging)
    brightness_score: float = 1.0
    glare_score: float = 1.0
    shadow_score: float = 1.0
    uniformity_score: float = 1.0


@dataclass
class IlluminationConfig:
    """Configuration for illumination processing."""
    # Gamma correction
    target_brightness: float = 128.0   # Target mean brightness (0-255)
    gamma_min: float = 0.5             # Minimum gamma value
    gamma_max: float = 2.0             # Maximum gamma value
    
    # CLAHE settings
    clahe_clip_limit: float = 2.0      # Contrast limiting threshold
    clahe_grid_size: Tuple[int, int] = (8, 8)  # Tile grid size
    
    # Shadow suppression
    bilateral_d: int = 9              # Diameter for bilateral filter
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    shadow_blend_alpha: float = 0.7   # Blend factor (0=original, 1=filtered)
    
    # Quality thresholds
    brightness_low_threshold: float = 40.0    # Below this = too dark
    brightness_high_threshold: float = 220.0  # Above this = overexposed
    glare_saturation_threshold: int = 250     # Pixel value for glare detection
    glare_ratio_threshold: float = 0.15       # Ratio of glare pixels
    shadow_threshold: int = 30                # Pixel value for shadow detection
    shadow_ratio_threshold: float = 0.25      # Ratio of shadow pixels
    
    # Anomaly thresholds
    anomaly_quality_threshold: float = 0.3    # Below this = flag as anomaly


class IlluminationProcessor:
    """
    Processes frames to normalize illumination conditions.
    
    Pipeline:
    1. Adaptive gamma correction to normalize overall brightness
    2. CLAHE on luminance channel for local contrast
    3. Mild shadow suppression with edge-preserving smoothing
    4. Compute quality score for temporal voting weights
    """
    
    def __init__(self, config: Optional[IlluminationConfig] = None):
        self.config = config or IlluminationConfig()
        
        # Pre-create CLAHE object for efficiency
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        
        # Pre-compute gamma lookup tables for common values
        self._gamma_luts = {}
    
    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        """Get or compute gamma correction lookup table."""
        # Round gamma to 2 decimal places for caching
        gamma_key = round(gamma, 2)
        
        if gamma_key not in self._gamma_luts:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype(np.uint8)
            self._gamma_luts[gamma_key] = table
        
        return self._gamma_luts[gamma_key]
    
    def _compute_adaptive_gamma(self, frame: np.ndarray) -> float:
        """
        Compute adaptive gamma value based on frame brightness.
        
        Dark frames get gamma < 1 (brightening)
        Bright frames get gamma > 1 (darkening)
        """
        # Convert to grayscale for brightness estimation
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Use mean brightness
        mean_brightness = np.mean(gray)
        
        # Compute gamma to bring mean towards target
        if mean_brightness < 1:
            mean_brightness = 1  # Avoid division by zero
        
        # Gamma calculation: target = current ^ (1/gamma)
        # So: gamma = log(target) / log(current)
        # Simplified: gamma = target / current (linear approximation)
        gamma = self.config.target_brightness / mean_brightness
        
        # Clamp gamma to reasonable range
        gamma = np.clip(gamma, self.config.gamma_min, self.config.gamma_max)
        
        return gamma
    
    def _apply_gamma_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive gamma correction to the frame."""
        gamma = self._compute_adaptive_gamma(frame)
        
        # Skip if gamma is close to 1.0 (no correction needed)
        if 0.95 <= gamma <= 1.05:
            return frame
        
        lut = self._get_gamma_lut(gamma)
        return cv2.LUT(frame, lut)
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the luminance channel only.
        
        Preserves color while enhancing local contrast.
        """
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split into channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to luminance channel
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _apply_shadow_suppression(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving shadow suppression.
        
        Uses bilateral filtering to reduce shadow intensity gradients
        while preserving door edges and texture.
        """
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            frame,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )
        
        # Blend with original to preserve some texture
        alpha = self.config.shadow_blend_alpha
        blended = cv2.addWeighted(
            frame, 1 - alpha,
            filtered, alpha,
            0
        )
        
        return blended
    
    def _compute_brightness_score(self, gray: np.ndarray) -> float:
        """
        Compute brightness quality score.
        
        Returns 1.0 for well-lit frames, lower for too dark or overexposed.
        """
        mean_brightness = np.mean(gray)
        
        low = self.config.brightness_low_threshold
        high = self.config.brightness_high_threshold
        target = self.config.target_brightness
        
        if mean_brightness < low:
            # Too dark - linear penalty
            return max(0.0, mean_brightness / low)
        elif mean_brightness > high:
            # Overexposed - linear penalty
            return max(0.0, (255 - mean_brightness) / (255 - high))
        else:
            # Good range - compute distance from optimal
            if mean_brightness <= target:
                return 0.8 + 0.2 * (mean_brightness - low) / (target - low)
            else:
                return 0.8 + 0.2 * (high - mean_brightness) / (high - target)
    
    def _compute_glare_score(self, gray: np.ndarray) -> float:
        """
        Compute glare quality score.
        
        Detects saturated/overexposed regions that indicate glare.
        Returns 1.0 for no glare, lower for significant glare.
        """
        threshold = self.config.glare_saturation_threshold
        max_ratio = self.config.glare_ratio_threshold
        
        # Count saturated pixels
        glare_pixels = np.sum(gray >= threshold)
        total_pixels = gray.size
        glare_ratio = glare_pixels / total_pixels
        
        if glare_ratio >= max_ratio:
            return 0.0
        else:
            # Linear decrease as glare increases
            return 1.0 - (glare_ratio / max_ratio)
    
    def _compute_shadow_score(self, gray: np.ndarray) -> float:
        """
        Compute shadow quality score.
        
        Detects large dark regions indicating deep shadows.
        Returns 1.0 for no shadows, lower for significant shadows.
        """
        threshold = self.config.shadow_threshold
        max_ratio = self.config.shadow_ratio_threshold
        
        # Count very dark pixels
        shadow_pixels = np.sum(gray <= threshold)
        total_pixels = gray.size
        shadow_ratio = shadow_pixels / total_pixels
        
        if shadow_ratio >= max_ratio:
            return 0.0
        else:
            # Linear decrease as shadows increase
            return 1.0 - (shadow_ratio / max_ratio)
    
    def _compute_uniformity_score(self, gray: np.ndarray) -> float:
        """
        Compute uniformity quality score.
        
        High variance indicates non-uniform lighting.
        Returns 1.0 for uniform lighting, lower for non-uniform.
        """
        # Use local variance as a measure of uniformity
        # High local variance in brightness indicates lighting issues
        std_dev = np.std(gray)
        
        # Normalize: typical std for good lighting is 40-60
        # Very high (>100) or very low (<20) indicates issues
        if std_dev < 20:
            # Very flat - might be overexposed or in shadow
            return 0.7
        elif std_dev > 100:
            # Very variable - lighting issues
            return max(0.3, 1.0 - (std_dev - 100) / 100)
        else:
            return 1.0
    
    def compute_quality_score(
        self, 
        frame: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute overall illumination quality score for a frame.
        
        Returns:
            (overall_score, brightness_score, glare_score, shadow_score, uniformity_score)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute individual scores
        brightness_score = self._compute_brightness_score(gray)
        glare_score = self._compute_glare_score(gray)
        shadow_score = self._compute_shadow_score(gray)
        uniformity_score = self._compute_uniformity_score(gray)
        
        # Weighted combination
        # Glare and shadows are most impactful for classification
        weights = {
            'brightness': 0.2,
            'glare': 0.35,
            'shadow': 0.35,
            'uniformity': 0.1
        }
        
        overall = (
            weights['brightness'] * brightness_score +
            weights['glare'] * glare_score +
            weights['shadow'] * shadow_score +
            weights['uniformity'] * uniformity_score
        )
        
        return overall, brightness_score, glare_score, shadow_score, uniformity_score
    
    def process_frame(self, frame: np.ndarray) -> IlluminationResult:
        """
        Process a single frame through the illumination normalization pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            IlluminationResult with normalized frame and quality metrics
        """
        if frame is None or frame.size == 0:
            return IlluminationResult(
                normalized_frame=frame,
                quality_score=0.0,
                is_anomaly=True,
                anomaly_reason="Empty or None frame"
            )
        
        # Step 1: Compute quality score on original frame
        (overall_score, brightness_score, glare_score, 
         shadow_score, uniformity_score) = self.compute_quality_score(frame)
        
        # Step 2: Apply adaptive gamma correction
        gamma_corrected = self._apply_gamma_correction(frame)
        
        # Step 3: Apply CLAHE on luminance channel
        clahe_enhanced = self._apply_clahe(gamma_corrected)
        
        # Step 4: Apply mild shadow suppression
        # Only apply if significant shadows detected
        if shadow_score < 0.7:
            normalized = self._apply_shadow_suppression(clahe_enhanced)
        else:
            normalized = clahe_enhanced
        
        # Determine if this is an anomaly
        is_anomaly = overall_score < self.config.anomaly_quality_threshold
        anomaly_reason = None
        
        if is_anomaly:
            reasons = []
            if brightness_score < 0.3:
                reasons.append("extreme brightness issue")
            if glare_score < 0.3:
                reasons.append("significant glare detected")
            if shadow_score < 0.3:
                reasons.append("deep shadows present")
            anomaly_reason = ", ".join(reasons) if reasons else "low overall quality"
        
        return IlluminationResult(
            normalized_frame=normalized,
            quality_score=overall_score,
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
            brightness_score=brightness_score,
            glare_score=glare_score,
            shadow_score=shadow_score,
            uniformity_score=uniformity_score
        )


def get_default_processor() -> IlluminationProcessor:
    """Get illumination processor with default configuration."""
    return IlluminationProcessor()


def get_conservative_processor() -> IlluminationProcessor:
    """Get processor with more conservative (less aggressive) settings."""
    config = IlluminationConfig(
        clahe_clip_limit=1.5,
        shadow_blend_alpha=0.5,
        gamma_min=0.7,
        gamma_max=1.5
    )
    return IlluminationProcessor(config)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Illumination Processor Module")
    print("=" * 50)
    
    # Test with a sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        frame = cv2.imread(image_path)
        
        if frame is not None:
            processor = IlluminationProcessor()
            result = processor.process_frame(frame)
            
            print(f"Quality Score: {result.quality_score:.3f}")
            print(f"  Brightness: {result.brightness_score:.3f}")
            print(f"  Glare: {result.glare_score:.3f}")
            print(f"  Shadow: {result.shadow_score:.3f}")
            print(f"  Uniformity: {result.uniformity_score:.3f}")
            print(f"Is Anomaly: {result.is_anomaly}")
            if result.anomaly_reason:
                print(f"Reason: {result.anomaly_reason}")
            
            # Save comparison
            comparison = np.hstack([frame, result.normalized_frame])
            output_path = image_path.replace(".", "_comparison.")
            cv2.imwrite(output_path, comparison)
            print(f"Saved comparison to: {output_path}")
        else:
            print(f"Could not read image: {image_path}")
    else:
        print("Usage: python illumination_processor.py <image_path>")
        print("\nModule loaded successfully!")
        
        # Quick self-test
        processor = IlluminationProcessor()
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = processor.process_frame(test_frame)
        print(f"Self-test quality score: {result.quality_score:.3f}")
