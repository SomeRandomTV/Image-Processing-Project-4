# Card Detection System

A computer vision system for detecting and classifying playing cards using OpenCV and template matching. This system can identify standard playing cards from images or webcam input by analyzing card rank and suit through corner detection and template matching algorithms.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a playing card detection and classification system that can:
- Capture cards from a webcam or load from image files
- Automatically correct card orientation and perspective distortion
- Extract rank and suit information from card corners
- Match against template images to identify the card
- Display intermediate processing steps and confidence metrics

## Features

- **Dual Input Modes**: Webcam capture or file-based image loading
- **Automatic Perspective Correction**: Uses contour detection and homography to flatten card images
- **Template Matching**: Compares extracted card features against a library of template images
- **Multi-Template Support**: Loads all template variations from directories (e.g., Ace0.jpg, Ace1.jpg, etc.)
- **Confidence Scoring**: Provides difference metrics to evaluate match quality
- **Visual Debugging**: Displays difference maps showing pixel-wise matching results
- **Grayscale Processing**: Optimized for performance and lighting invariance

## System Architecture

The system consists of three main components:

### 1. Card Rotator (`card_rotator.py`)
Handles geometric transformations to normalize card orientation:
- Thresholding to isolate the card from background
- Contour detection to find card boundaries
- Perspective transformation to correct viewing angle
- Cropping to remove excess background

### 2. Card Detector (`card_detector.py`)
Performs feature extraction and template matching:
- Corner region extraction (top-left 32x84 pixels)
- Rank and suit isolation through thresholding and contour analysis
- Resizing symbols to standardized dimensions (70x125 for ranks, 70x100 for suits)
- Template matching using pixel-wise absolute difference
- Confidence thresholding to filter poor matches

### 3. Main Application (`main.py`)
Provides user interface and orchestrates the detection pipeline:
- User input handling (webcam or file selection)
- Pipeline coordination (rotation, detection, classification)
- Results display and visualization
- Error handling and user feedback

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Packages

```bash
pip install opencv-python numpy
```

### Optional Packages

```bash
pip install matplotlib  # For intermediate step visualization
```

### Setup Template Images

Create two directories in the project root:

```
CardDetector/
├── RANKS/
│   ├── Ace0.jpg
│   ├── Ace1.jpg
│   ├── Two0.jpg
│   └── ...
└── SUITS/
    ├── Spades0.jpg
    ├── Hearts0.jpg
    ├── Diamonds0.jpg
    └── Clubs0.jpg
```

Template images should be:
- Grayscale or color (will be converted to grayscale)
- Clear, high-contrast images of card ranks and suits
- Named with pattern: `<Name><Number>.jpg` (e.g., King0.jpg, Spades2.jpg)
- Numbers are stripped from output (King0 displays as "King")

## Usage

### Running the Application

```bash
python main.py
```

### Option 1: Webcam Capture

1. Select option `1` from the menu
2. Position the card in front of the webcam
3. Press `SPACE` to capture the image
4. Press `ESC` to cancel

**Tips for Best Results:**
- Use good lighting with minimal shadows
- Hold the card against a contrasting background
- Ensure the entire card is visible in frame
- Avoid glare or reflections on the card surface

### Option 2: File Input

1. Select option `2` from the menu
2. Enter the full path to the image file
3. The system accepts standard image formats (JPG, PNG, BMP, etc.)

**Example:**
```
Enter the path to the card image: /Users/username/Desktop/card_image.jpg
```

### Understanding the Output

The system displays several windows:

1. **Original Image**: The raw input image
2. **Processed Card**: The card after rotation and perspective correction
3. **Detected Rank**: The extracted rank symbol (70x125 pixels)
4. **Detected Suit**: The extracted suit symbol (70x100 pixels)
5. **Rank Difference Map**: Heatmap showing matching quality for rank (blue = good, red = poor)
6. **Suit Difference Map**: Heatmap showing matching quality for suit

### Console Output

```
[1/3] Applying rotation and perspective transformation...
   ✓ Card rotated and flattened (size: (400, 300))

[2/3] Detecting and classifying card...
   ✓ Detection complete

[3/3] Results:
==================================================
   Card Identified: Jack of Spades
   Confidence: Jack of Spades (rank_diff: 850, suit_diff: 320)
==================================================
```

**Confidence Metrics:**
- Lower numbers indicate better matches
- Rank threshold: 2000 (values above this result in "Unknown")
- Suit threshold: 700 (values above this result in "Unknown")

## How It Works

### Step 1: Image Preprocessing

```
Input Image → Grayscale Conversion → Thresholding → Contour Detection
```

The system converts the image to grayscale and applies adaptive thresholding to isolate the card from the background.

### Step 2: Perspective Correction

```
Contour → Corner Detection → Homography Matrix → Warped Image
```

The largest contour is identified as the card boundary. The four corners are used to compute a perspective transformation that creates a top-down view of the card.

### Step 3: Corner Extraction

```
Top-Left Corner (32x84) → 4x Zoom → Adaptive Thresholding
```

The top-left corner of the card is extracted and zoomed for better detail. Adaptive thresholding isolates the rank and suit symbols.

### Step 4: Symbol Isolation

```
Corner Region → Split (Rank/Suit) → Contour Detection → Resize to Standard Size
```

The corner is split into rank (top) and suit (bottom) regions. The largest contour in each region is extracted and resized to standardized dimensions.

### Step 5: Template Matching

```
For each template:
    Difference = Sum(|ExtractedImage - Template|) / 255
    
Best Match = Template with minimum Difference
```

The system compares the extracted symbols against all templates using pixel-wise absolute difference. The template with the lowest difference score is selected.

### Step 6: Classification

```
If Difference < Threshold:
    Return Matched Name (with numbers stripped)
Else:
    Return "Unknown"
```

The best match is returned if its confidence score is below the threshold. Template names have numbers stripped for clean output (e.g., "King0" becomes "King").

## File Structure

```
CardDetector/
├── main.py                 # Main application entry point
├── card_detector.py        # Detection and classification logic
├── card_rotator.py         # Geometric transformation utilities
├── RANKS/                  # Rank template images
│   ├── Ace0.jpg
│   ├── Ace1.jpg
│   └── ...
├── SUITS/                  # Suit template images
│   ├── Spades0.jpg
│   └── ...
├── Card_Imgs/             # (Optional) Sample card images
├── Testimages/            # (Optional) Test images
└── README.md              # This file
```

## Configuration

### Adjusting Detection Parameters

In `card_detector.py`, you can modify:

```python
# Corner dimensions (pixels)
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Standardized symbol dimensions
RANK_WIDTH = 70
RANK_HEIGHT = 125
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# Matching thresholds (higher = more lenient)
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

# Threshold offset for corner isolation
CARD_THRESH = 30
```

**When to Adjust:**
- **RANK_DIFF_MAX / SUIT_DIFF_MAX**: If the system is too strict (rejects valid cards) or too lenient (accepts incorrect matches)
- **CARD_THRESH**: If corner extraction fails due to lighting variations
- **CORNER_WIDTH / CORNER_HEIGHT**: If using non-standard card sizes

### Adding New Templates

1. Capture or create clear images of card ranks/suits
2. Save as JPG files in RANKS/ or SUITS/ directories
3. Use naming pattern: `<Name><Number>.jpg`
4. No code changes needed - templates are loaded automatically

**Example:**
```
RANKS/Ace0.jpg      # First Ace template
RANKS/Ace1.jpg      # Second Ace template (different lighting/angle)
RANKS/King0.jpg     # First King template
SUITS/Hearts0.jpg   # First Hearts template
```

## Troubleshooting

### Card Not Detected

**Symptoms:** "Unknown" result for rank or suit

**Solutions:**
- Ensure good lighting without harsh shadows
- Verify template images exist in RANKS/ and SUITS/ directories
- Check that template images are clear and high-contrast
- Try adjusting RANK_DIFF_MAX and SUIT_DIFF_MAX thresholds
- Examine difference maps for mismatch patterns

### Perspective Correction Fails

**Symptoms:** Card appears distorted or cropped incorrectly

**Solutions:**
- Ensure the entire card is visible in the input image
- Use a contrasting background (dark card on light surface or vice versa)
- Verify the card edges are clearly defined
- Check that the image is not too blurry or low-resolution

### Templates Not Loading

**Symptoms:** "Warning: Could not load rank/suit image" messages

**Solutions:**
- Verify RANKS/ and SUITS/ directories exist
- Check file permissions (read access required)
- Ensure image files are valid JPG/PNG format
- Confirm filenames don't contain special characters

### Webcam Access Denied

**Symptoms:** "Error: Could not open webcam"

**Solutions:**
- Grant camera permissions to Terminal/Python in System Preferences (macOS)
- Verify camera is not in use by another application
- Try a different camera index: `cv.VideoCapture(1)` instead of `(0)`

### Performance Issues

**Solutions:**
- Reduce input image resolution
- Minimize number of template images
- Use grayscale input images
- Close other CPU-intensive applications

## Algorithm Complexity

- **Template Matching**: O(T × W × H) where T = number of templates, W × H = image dimensions
- **Contour Detection**: O(N) where N = number of pixels
- **Perspective Transform**: O(W × H)

**Typical Processing Time:**
- Card rotation and perspective correction: 50-100ms
- Template matching: 10-50ms (depends on number of templates)
- Total pipeline: 100-200ms on modern hardware

## Limitations

- Requires standard playing card layout (rank and suit in top-left corner)
- Sensitive to lighting conditions (shadows can affect thresholding)
- Works best with solid, contrasting backgrounds
- Template matching is scale-sensitive (card must be similar size to templates)
- Does not handle overlapping or partially occluded cards
- Limited to grayscale comparison (color information not used)

## Future Enhancements

Potential improvements:
- Multi-scale template matching for size invariance
- Feature-based matching (SIFT/ORB) for rotation invariance
- Deep learning classification for improved accuracy
- Batch processing mode for multiple cards
- Real-time video stream processing
- Color-based suit detection for improved reliability

## License

This project is provided as-is for educational purposes.

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## Contact

For questions or issues, please open an issue on the repository.
