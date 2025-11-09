import os
import cv2 as cv
import numpy as np
from typing import Tuple, List, Optional


class CardDetector:
    """
    Card detection and classification using template matching.
    This module handles the detection of card rank and suit from preprocessed card images.
    """

    # Thresholding constants
    CARD_THRESH = 30  # Threshold offset from white level for card corner

    # Corner dimensions where rank and suit are located
    CORNER_WIDTH = 32
    CORNER_HEIGHT = 84

    # Dimensions for standardized rank and suit images
    RANK_WIDTH = 70
    RANK_HEIGHT = 125
    SUIT_WIDTH = 70
    SUIT_HEIGHT = 100

    # Maximum difference thresholds for matching
    RANK_DIFF_MAX = 2000
    SUIT_DIFF_MAX = 700

    def __init__(self, ranks_path: str, suits_path: str):
        """
        Initialize the CardDetector with training images for template matching.
        
        Args:
            ranks_path: Path to directory containing rank template images
            suits_path: Path to directory containing suit template images
        """
        self.ranks_path = ranks_path
        self.suits_path = suits_path
        self.train_ranks = self._load_ranks()
        self.train_suits = self._load_suits()
        self._intermediate_steps = None

    # ---------------- Training Image Loading ----------------

    def _load_ranks(self) -> List['TrainRank']:
        """
        Load all rank template images from the RANKS directory.
        Loads any image file regardless of name.
        
        Returns:
            List of TrainRank objects containing template images and names
        """
        train_ranks = []
        
        if not os.path.exists(self.ranks_path):
            print(f"Warning: Ranks directory not found: {self.ranks_path}")
            return train_ranks
        
        # Get all files in the directory
        for filename in sorted(os.listdir(self.ranks_path)):
            filepath = os.path.join(self.ranks_path, filename)
            
            # Skip directories and hidden files
            if os.path.isdir(filepath) or filename.startswith('.'):
                continue
            
            # Try to load as image
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            if img is not None:
                # Use filename without extension as the name
                name = os.path.splitext(filename)[0]
                train_ranks.append(TrainRank(name, img))
                print(f"Loaded rank: {filename}")
            else:
                print(f"Warning: Could not load as image: {filename}")
        
        print(f"Total ranks loaded: {len(train_ranks)}")
        return train_ranks

    def _load_suits(self) -> List['TrainSuit']:
        """
        Load all suit template images from the SUITS directory.
        Loads any image file regardless of name.
        
        Returns:
            List of TrainSuit objects containing template images and names
        """
        train_suits = []
        
        if not os.path.exists(self.suits_path):
            print(f"Warning: Suits directory not found: {self.suits_path}")
            return train_suits
        
        # Get all files in the directory
        for filename in sorted(os.listdir(self.suits_path)):
            filepath = os.path.join(self.suits_path, filename)
            
            # Skip directories and hidden files
            if os.path.isdir(filepath) or filename.startswith('.'):
                continue
            
            # Try to load as image
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            if img is not None:
                # Use filename without extension as the name
                name = os.path.splitext(filename)[0]
                train_suits.append(TrainSuit(name, img))
                print(f"Loaded suit: {filename}")
            else:
                print(f"Warning: Could not load as image: {filename}")
        
        print(f"Total suits loaded: {len(train_suits)}")
        return train_suits

    # ---------------- Card Corner Isolation ----------------

    def isolate_rank_and_suit(self, card_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract and isolate the rank and suit from a preprocessed card image.
        Expects a vertical, upright card image from CardRotator.
        
        Args:
            card_image: Preprocessed card image (grayscale, vertical orientation)
            
        Returns:
            Tuple of (rank_image, suit_image) - standardized, thresholded images
            Returns (None, None) if extraction fails
        """
        # Ensure card is vertical (taller than wide)
        # If it's horizontal, rotate it 90 degrees
        h, w = card_image.shape[:2]
        if w > h:
            card_image = cv.rotate(card_image, cv.ROTATE_90_CLOCKWISE)
            h, w = card_image.shape[:2]
        
        # Extract the top-left corner where rank and suit are located
        corner = card_image[0:self.CORNER_HEIGHT, 0:self.CORNER_WIDTH]
        
        # Zoom 4x on the corner for better detail
        corner_zoom = cv.resize(corner, (0, 0), fx=4, fy=4)
        
        # Sample a white pixel to determine threshold level
        # This makes it adaptive to different card designs
        white_level = int(corner_zoom[15, int((self.CORNER_WIDTH * 4) / 2)])
        
        # Calculate threshold level with bounds checking to prevent overflow
        thresh_level = max(1, white_level - self.CARD_THRESH)
        
        # Threshold to isolate rank and suit (white background, black symbols)
        _, corner_thresh = cv.threshold(corner_zoom, thresh_level, 255, cv.THRESH_BINARY_INV)
        
        # Split into top half (rank) and bottom half (suit)
        rank_region = corner_thresh[20:185, 0:128]
        suit_region = corner_thresh[186:336, 0:128]
        
        # Extract rank
        rank_img = self._extract_and_resize_symbol(rank_region, self.RANK_WIDTH, self.RANK_HEIGHT)
        
        # Extract suit
        suit_img = self._extract_and_resize_symbol(suit_region, self.SUIT_WIDTH, self.SUIT_HEIGHT)
        
        # Store intermediate steps for visualization
        self._intermediate_steps = {
            'corner': corner,
            'corner_zoom': corner_zoom,
            'corner_thresh': corner_thresh,
            'rank_region': rank_region,
            'suit_region': suit_region,
            'thresh_level': thresh_level
        }
        
        return rank_img, suit_img

    @staticmethod
    def _extract_and_resize_symbol(region: np.ndarray, target_width: int, target_height: int) -> Optional[np.ndarray]:
        """
        Find the largest contour in a region and resize it to target dimensions.
        
        Args:
            region: Thresholded image region containing the symbol
            target_width: Desired width of output
            target_height: Desired height of output
            
        Returns:
            Resized symbol image, or None if no contour found
        """
        contours, _ = cv.findContours(region, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Sort contours by area and get the largest
        contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)
        
        # Get bounding rectangle of largest contour
        x, y, w, h = cv.boundingRect(contours_sorted[0])
        
        # Extract and resize the symbol
        symbol_roi = region[y:y+h, x:x+w]
        symbol_resized = cv.resize(symbol_roi, (target_width, target_height))
        
        return symbol_resized

    # ---------------- Template Matching ----------------

    def match_card(self, rank_img: Optional[np.ndarray], suit_img: Optional[np.ndarray]) -> 'CardIdentity':
        """
        Identify the card by comparing rank and suit images to training templates.
        Uses pixel-wise difference to find the best match.
        
        Args:
            rank_img: Extracted and standardized rank image
            suit_img: Extracted and standardized suit image
            
        Returns:
            CardIdentity object containing the matched rank, suit, and confidence scores
        """
        best_rank_match = "Unknown"
        best_suit_match = "Unknown"
        best_rank_diff = 10000
        best_suit_diff = 10000
        best_rank_diff_img = None
        best_suit_diff_img = None
        
        # Only attempt matching if both images were successfully extracted
        if rank_img is not None and suit_img is not None:
            # Match rank
            for train_rank in self.train_ranks:
                # Calculate absolute difference between query and training image
                diff_img = cv.absdiff(rank_img, train_rank.img)
                # Sum all differences and normalize by dividing by 255
                rank_diff = int(np.sum(diff_img) / 255)
                
                if rank_diff < best_rank_diff:
                    best_rank_diff = rank_diff
                    best_rank_match = train_rank.name
                    best_rank_diff_img = diff_img.copy()
            
            # Match suit
            for train_suit in self.train_suits:
                diff_img = cv.absdiff(suit_img, train_suit.img)
                suit_diff = int(np.sum(diff_img) / 255)
                
                if suit_diff < best_suit_diff:
                    best_suit_diff = suit_diff
                    best_suit_match = train_suit.name
                    best_suit_diff_img = diff_img.copy()
        
        # Apply confidence thresholds - if difference is too high, mark as Unknown
        if best_rank_diff > self.RANK_DIFF_MAX:
            best_rank_match = "Unknown"
        
        if best_suit_diff > self.SUIT_DIFF_MAX:
            best_suit_match = "Unknown"

        # Clean up names: extract the base card name (before any underscore or trailing digits)
        if best_rank_match != "Unknown":
            # Split on underscore and take first part (e.g., "Eight_webcam_capture_123" -> "Eight")
            best_rank_match = best_rank_match.split('_')[0]
            # Then strip trailing digits (e.g., "Eight0" -> "Eight")
            best_rank_match = best_rank_match.rstrip('0123456789')
        
        if best_suit_match != "Unknown":
            # Split on underscore and take first part
            best_suit_match = best_suit_match.split('_')[0]
            # Then strip trailing digits
            best_suit_match = best_suit_match.rstrip('0123456789')
        
        return CardIdentity(
            rank=best_rank_match,
            suit=best_suit_match,
            rank_confidence=best_rank_diff,
            suit_confidence=best_suit_diff,
            rank_diff_img=best_rank_diff_img,
            suit_diff_img=best_suit_diff_img
        )

    # ---------------- Complete Detection Pipeline ----------------

    def detect_and_classify(self, card_image: np.ndarray) -> 'CardIdentity':
        """
        Complete pipeline: Extract rank/suit from card image and classify it.
        Expects card from CardRotator (should be vertical/upright).
        
        Args:
            card_image: Preprocessed card image from CardRotator (grayscale)
            
        Returns:
            CardIdentity object with detected card information
        """
        rank_img, suit_img = self.isolate_rank_and_suit(card_image)
        card_identity = self.match_card(rank_img, suit_img)
        
        # Store the extracted images for debugging/visualization
        card_identity.rank_img = rank_img
        card_identity.suit_img = suit_img
        
        return card_identity

    # ---------------- Visualization ----------------

    def visualize_intermediate_steps(self, card_identity: 'CardIdentity') -> None:
        """
        Display all intermediate processing steps using matplotlib.
        Shows the complete pipeline from corner extraction to final classification.
        
        Args:
            card_identity: CardIdentity object from detect_and_classify
        """
        if self._intermediate_steps is None:
            print("No intermediate steps available. Run detect_and_classify first.")
            return
        
        # Import matplotlib only when visualization is needed
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is required for visualization. Install it with: pip install matplotlib")
            return
        
        steps = self._intermediate_steps
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Card Detection Pipeline - Intermediate Steps', fontsize=16, fontweight='bold')
        
        # Row 1: Corner extraction and zooming
        plt.subplot(3, 4, 1)
        plt.imshow(steps['corner'], cmap='gray')
        plt.title(f'1. Corner Extraction\n({self.CORNER_WIDTH}x{self.CORNER_HEIGHT})')
        plt.axis('off')
        
        plt.subplot(3, 4, 2)
        plt.imshow(steps['corner_zoom'], cmap='gray')
        plt.title('2. Corner Zoom (4x)')
        plt.axis('off')
        
        plt.subplot(3, 4, 3)
        plt.imshow(steps['corner_thresh'], cmap='gray')
        plt.title(f'3. Thresholded\n(threshold={steps["thresh_level"]})')
        plt.axis('off')
        
        # Show rank and suit regions
        plt.subplot(3, 4, 4)
        # Combine rank and suit regions for visualization
        combined = np.vstack([steps['rank_region'], 
                             np.zeros((10, steps['rank_region'].shape[1]), dtype=np.uint8),
                             steps['suit_region']])
        plt.imshow(combined, cmap='gray')
        plt.title('4. Split Regions\n(Rank + Suit)')
        plt.axis('off')
        
        # Row 2: Extracted and resized symbols
        plt.subplot(3, 4, 5)
        if card_identity.rank_img is not None:
            plt.imshow(card_identity.rank_img, cmap='gray')
            plt.title(f'5. Extracted Rank\n({self.RANK_WIDTH}x{self.RANK_HEIGHT})')
        else:
            plt.text(0.5, 0.5, 'No Rank\nExtracted', ha='center', va='center')
            plt.title('5. Extracted Rank')
        plt.axis('off')
        
        plt.subplot(3, 4, 6)
        if card_identity.suit_img is not None:
            plt.imshow(card_identity.suit_img, cmap='gray')
            plt.title(f'6. Extracted Suit\n({self.SUIT_WIDTH}x{self.SUIT_HEIGHT})')
        else:
            plt.text(0.5, 0.5, 'No Suit\nExtracted', ha='center', va='center')
            plt.title('6. Extracted Suit')
        plt.axis('off')
        
        # Show best matching templates
        plt.subplot(3, 4, 7)
        if card_identity.rank != "Unknown" and len(self.train_ranks) > 0:
            # Find the matching template
            for train_rank in self.train_ranks:
                if train_rank.name == card_identity.rank:
                    plt.imshow(train_rank.img, cmap='gray')
                    break
            plt.title(f'7. Best Rank Match\n"{card_identity.rank}"')
        else:
            plt.text(0.5, 0.5, 'Unknown', ha='center', va='center')
            plt.title('7. Best Rank Match')
        plt.axis('off')
        
        plt.subplot(3, 4, 8)
        if card_identity.suit != "Unknown" and len(self.train_suits) > 0:
            # Find the matching template
            for train_suit in self.train_suits:
                if train_suit.name == card_identity.suit:
                    plt.imshow(train_suit.img, cmap='gray')
                    break
            plt.title(f'8. Best Suit Match\n"{card_identity.suit}"')
        else:
            plt.text(0.5, 0.5, 'Unknown', ha='center', va='center')
            plt.title('8. Best Suit Match')
        plt.axis('off')
        
        # Row 3: Difference maps
        plt.subplot(3, 4, 9)
        if card_identity.rank_diff_img is not None:
            plt.imshow(card_identity.rank_diff_img, cmap='hot')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f'9. Rank Diff Map\n(diff={card_identity.rank_confidence})')
        else:
            plt.text(0.5, 0.5, 'No Diff Map', ha='center', va='center')
            plt.title('9. Rank Diff Map')
        plt.axis('off')
        
        plt.subplot(3, 4, 10)
        if card_identity.suit_diff_img is not None:
            plt.imshow(card_identity.suit_diff_img, cmap='hot')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f'10. Suit Diff Map\n(diff={card_identity.suit_confidence})')
        else:
            plt.text(0.5, 0.5, 'No Diff Map', ha='center', va='center')
            plt.title('10. Suit Diff Map')
        plt.axis('off')
        
        # Final result summary
        plt.subplot(3, 4, 11)
        plt.axis('off')
        result_text = f"Final Result:\n\n"
        result_text += f"Rank: {card_identity.rank}\n"
        result_text += f"Rank Confidence: {card_identity.rank_confidence}\n"
        result_text += f"Max Allowed: {self.RANK_DIFF_MAX}\n\n"
        result_text += f"Suit: {card_identity.suit}\n"
        result_text += f"Suit Confidence: {card_identity.suit_confidence}\n"
        result_text += f"Max Allowed: {self.SUIT_DIFF_MAX}\n\n"
        
        if card_identity.is_valid():
            result_text += f"\n✓ IDENTIFIED:\n{card_identity}"
            color = 'green'
        else:
            result_text += f"\n✗ UNKNOWN"
            color = 'red'
        
        plt.text(0.1, 0.5, result_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        plt.title('11. Classification Results')
        
        plt.subplot(3, 4, 12)
        plt.axis('off')
        legend_text = "Legend:\n\n"
        legend_text += "Diff Map Colors:\n"
        legend_text += "• Dark (Black): Perfect match\n"
        legend_text += "• Bright (White/Yellow): Mismatch\n\n"
        legend_text += "Lower difference = Better match\n\n"
        legend_text += "Threshold:\n"
        legend_text += f"• Rank < {self.RANK_DIFF_MAX}\n"
        legend_text += f"• Suit < {self.SUIT_DIFF_MAX}"
        
        plt.text(0.1, 0.5, legend_text, fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        plt.title('12. Information')
        
        plt.tight_layout()
        
        # Save to file instead of showing interactively (avoids GUI issues)
        output_file = 'card_detection_steps.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_file}")
        print("  Opening the image...")
        
        # Open the image with the default viewer
        import subprocess
        subprocess.run(['open', output_file])  # macOS
        
        plt.close()


# ---------------- Data Classes ----------------

class TrainRank:
    """Container for training rank images."""
    def __init__(self, name: str, img: np.ndarray):
        self.name = name
        self.img = img


class TrainSuit:
    """Container for training suit images."""
    def __init__(self, name: str, img: np.ndarray):
        self.name = name
        self.img = img


class CardIdentity:
    """
    Container for card detection results.
    
    Attributes:
        rank: Detected rank name (e.g., "Ace", "King", "Unknown")
        suit: Detected suit name (e.g., "Spades", "Hearts", "Unknown")
        rank_confidence: Difference score for rank match (lower is better)
        suit_confidence: Difference score for suit match (lower is better)
        rank_img: Extracted rank image (for debugging)
        suit_img: Extracted suit image (for debugging)
        rank_diff_img: Difference map for best rank match (for visualization)
        suit_diff_img: Difference map for best suit match (for visualization)
    """
    def __init__(self, rank: str, suit: str, rank_confidence: int, suit_confidence: int,
                 rank_diff_img: Optional[np.ndarray] = None, suit_diff_img: Optional[np.ndarray] = None):
        self.rank = rank
        self.suit = suit
        self.rank_confidence = rank_confidence
        self.suit_confidence = suit_confidence
        self.rank_img: Optional[np.ndarray] = None
        self.suit_img: Optional[np.ndarray] = None
        self.rank_diff_img: Optional[np.ndarray] = rank_diff_img
        self.suit_diff_img: Optional[np.ndarray] = suit_diff_img
    
    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"
    
    def is_valid(self) -> bool:
        """Check if both rank and suit were successfully identified."""
        return self.rank != "Unknown" and self.suit != "Unknown"
    
    def get_confidence_info(self) -> str:
        """Get detailed confidence information for debugging."""
        return f"{self.rank} of {self.suit} (rank_diff: {self.rank_confidence}, suit_diff: {self.suit_confidence})"