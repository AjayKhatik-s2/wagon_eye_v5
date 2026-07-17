"""
Wagon Number Aggregator - Temporal aggregation of OCR results.

Aggregates WagonNumberResult objects across frames to produce
consistent wagon number assignments.

Key features:
- Groups results by exact wagon number string
- Tracks frame counts and confidence scores
- Returns best result per unique wagon number
- Links OCR results to wagon sequential numbers via spatial position
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AggregatorConfig:
    """Configuration for wagon number aggregation."""
    min_frame_count: int = 2         # Minimum detections to confirm
    majority_threshold: float = 0.5  # Fraction of frames with same number
    min_confidence: float = 0.3      # Minimum confidence to include
    spatial_merge_distance: float = 100.0  # Pixels to merge nearby detections
    require_validation: bool = True  # Only include validated numbers


@dataclass 
class AggregatedWagonNumber:
    """
    Aggregated wagon number with temporal consistency.
    """
    # Best wagon number string
    wagon_number: str = ""
    
    # Parsed fields
    wagon_type: str = ""
    owning_railway: str = ""
    year_of_manufacture: str = ""
    individual_number: str = ""
    check_digit: str = ""
    
    # Aggregation stats
    frame_count: int = 0
    confidence: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    
    # Spatial info for wagon assignment
    avg_center_x: float = 0.0
    
    # Assigned wagon sequential number (after finalization)
    wagon_index: int = -1
    
    @property
    def formatted_number(self) -> str:
        """Return formatted wagon number."""
        if len(self.wagon_number) == 11:
            return f"{self.wagon_type} {self.owning_railway} {self.year_of_manufacture} {self.individual_number} {self.check_digit}"
        return self.wagon_number if self.wagon_number else "-"
    
    def __repr__(self):
        return f"AggWagon({self.formatted_number}, frames={self.frame_count}, conf={self.confidence:.2f})"


class WagonNumberAggregator:
    """
    Aggregates OCR results across frames for temporal consistency.
    
    Groups detections by wagon number string and tracks:
    - Frame counts
    - Confidence scores
    - Spatial positions (for wagon assignment)
    """
    
    def __init__(self, config: AggregatorConfig = None):
        """
        Initialize aggregator.
        
        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregatorConfig()
        
        # Storage: wagon_number -> list of results
        self._results: Dict[str, List] = defaultdict(list)
        
        # Spatial tracking: center_x -> wagon_number
        self._spatial_map: Dict[int, str] = {}
    
    def reset(self):
        """Reset aggregator for new video."""
        self._results.clear()
        self._spatial_map.clear()
    
    def add_result(self, result) -> None:
        """
        Add a WagonNumberResult to aggregation (legacy API).
        
        Args:
            result: WagonNumberResult from detector
        """
        # Skip if validation required and result is invalid
        if self.config.require_validation and not result.is_valid:
            return
        
        # Skip low confidence
        if result.confidence < self.config.min_confidence:
            return
        
        # Store result keyed by wagon number
        wagon_number = result.wagon_number
        if wagon_number:
            self._results[wagon_number].append({
                'result': result,
                'frame_idx': result.frame_idx,
                'center_x': getattr(result, 'center_x', 0),
                'confidence': result.confidence
            })
    
    def add_wagon_number(self, wagon_num, frame_idx: int) -> None:
        """
        Add a WagonNumber to aggregation (new API).
        
        Args:
            wagon_num: WagonNumber object from WagonNumberOCR
            frame_idx: Current frame index
        """
        # Skip if validation required and result is invalid
        if self.config.require_validation and not wagon_num.is_valid:
            return
        
        # Skip low confidence
        if wagon_num.ocr_confidence < self.config.min_confidence:
            return
        
        # Store result keyed by full wagon number
        number = wagon_num.full_number
        if number:
            self._results[number].append({
                'wagon_num': wagon_num,
                'frame_idx': frame_idx,
                'center_x': 0,  # Not tracked in new API
                'confidence': wagon_num.ocr_confidence
            })
    
    def _vote_for_best_digits(self, results: List) -> str:
        """
        Use digit-level voting to find the best wagon number.
        
        When OCR produces multiple readings with slight errors,
        voting on each digit position gives the most accurate result.
        
        Args:
            results: List of OCR result dicts
            
        Returns:
            Best wagon number string based on voting
        """
        if not results:
            return ""
        
        if len(results) == 1:
            # Handle both old and new API formats
            r = results[0].get('result') or results[0].get('wagon_num')
            return r.wagon_number if hasattr(r, 'wagon_number') else r.full_number
        
        # Get all wagon numbers - handle both formats
        wagon_numbers = []
        for r in results:
            obj = r.get('result') or r.get('wagon_num')
            if obj:
                num = getattr(obj, 'wagon_number', None) or getattr(obj, 'full_number', '')
                if num:
                    wagon_numbers.append(num)
        
        if not wagon_numbers:
            return ""
        
        # Only vote if all have same length
        lengths = set(len(n) for n in wagon_numbers)
        if len(lengths) != 1:
            # Return the most common full string
            from collections import Counter
            return Counter(wagon_numbers).most_common(1)[0][0]
        
        num_digits = list(lengths)[0]
        
        # Vote for each digit position
        from collections import Counter
        best_number = ""
        
        for pos in range(num_digits):
            digits_at_pos = [n[pos] for n in wagon_numbers if pos < len(n)]
            if digits_at_pos:
                # Get most common digit at this position
                counter = Counter(digits_at_pos)
                best_digit = counter.most_common(1)[0][0]
                best_number += best_digit
        
        return best_number
    
    def get_aggregated_numbers(self) -> List[AggregatedWagonNumber]:
        """
        Get aggregated wagon numbers meeting thresholds.
        
        Uses digit-level voting for accuracy when multiple OCR reads differ.
        
        Returns:
            List of AggregatedWagonNumber sorted by first frame
        """
        aggregated = []
        
        for wagon_number, results in self._results.items():
            if len(results) < self.config.min_frame_count:
                continue
            
            # Get best result (highest confidence) as base
            best = max(results, key=lambda r: r['confidence'])
            # Handle both old and new API formats
            best_result = best.get('result') or best.get('wagon_num')
            
            # Use digit voting if we have multiple results
            voted_number = self._vote_for_best_digits(results)
            final_number = voted_number if voted_number else wagon_number
            
            # Compute aggregated stats
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            avg_center_x = sum(r['center_x'] for r in results) / len(results)
            first_frame = min(r['frame_idx'] for r in results)
            last_frame = max(r['frame_idx'] for r in results)
            
            # Parse final number for fields
            if len(final_number) == 11:
                wt = final_number[0:2]
                orw = final_number[2:4]
                yom = final_number[4:6]
                inum = final_number[6:10]
                cd = final_number[10]
            else:
                wt = best_result.wagon_type
                orw = best_result.owning_railway
                yom = best_result.year_of_manufacture
                inum = best_result.individual_number
                cd = best_result.check_digit
            
            agg = AggregatedWagonNumber(
                wagon_number=final_number,
                wagon_type=wt,
                owning_railway=orw,
                year_of_manufacture=yom,
                individual_number=inum,
                check_digit=cd,
                frame_count=len(results),
                confidence=avg_confidence,
                first_frame=first_frame,
                last_frame=last_frame,
                avg_center_x=avg_center_x
            )
            
            aggregated.append(agg)
        
        # Sort by first frame (temporal order)
        aggregated.sort(key=lambda a: a.first_frame)
        
        # Assign wagon indices based on temporal order
        for idx, agg in enumerate(aggregated, start=1):
            agg.wagon_index = idx
        
        return aggregated
    
    def get_wagon_numbers_dict(self) -> Dict[str, AggregatedWagonNumber]:
        """
        Get wagon numbers as dict keyed by wagon number string.
        
        Returns:
            Dict mapping wagon_number -> AggregatedWagonNumber
        """
        aggregated = self.get_aggregated_numbers()
        return {agg.wagon_number: agg for agg in aggregated}
    
    def get_wagon_number_for_index(self, wagon_index: int) -> Optional[str]:
        """
        Get wagon number for a specific wagon index (1-indexed).
        
        Args:
            wagon_index: Wagon sequential number (1, 2, 3, ...)
            
        Returns:
            Formatted wagon number or None
        """
        aggregated = self.get_aggregated_numbers()
        
        # Find by index
        for agg in aggregated:
            if agg.wagon_index == wagon_index:
                return agg.formatted_number
        
        return None
    
    def get_stats(self) -> Dict:
        """Get aggregation statistics."""
        total_results = sum(len(v) for v in self._results.values())
        unique_numbers = len(self._results)
        confirmed = len(self.get_aggregated_numbers())
        
        return {
            'total_ocr_results': total_results,
            'unique_numbers': unique_numbers,
            'confirmed_numbers': confirmed
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Wagon Number Aggregator Module")
    print("This module aggregates OCR results for temporal consistency.")
    
    # Demo usage
    config = AggregatorConfig(
        min_frame_count=2,
        min_confidence=0.3,
        require_validation=True
    )
    
    aggregator = WagonNumberAggregator(config)
    print(f"Initialized with config: {config}")
