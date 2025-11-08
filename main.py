import cv2 as cv
import numpy as np
import os
import time
from card_rotator import CardRotator
from card_detector import CardDetector, CardIdentity


def get_user_choice() -> str:
    """
    Prompt user to choose between webcam or file input.
    
    Returns:
        'webcam' or 'file'
    """
    print("\n" + "="*50)
    print("Card Detection System")
    print("="*50)
    print("\nHow would you like to provide the card image?")
    print("1. Webcam (press SPACE to capture)")
    print("2. Read from file")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == '1':
            return 'webcam'
        elif choice == '2':
            return 'file'
        else:
            print("Invalid choice. Please enter 1 or 2.")


def capture_from_webcam():
    """
    Capture an image from the webcam.
    
    Returns:
        Captured image as numpy array, or None if capture fails
    """
    print("\nOpening webcam...")
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    print("Press SPACE to capture the card, or ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Display the frame
        cv.imshow('Webcam - Press SPACE to capture', frame)
        
        key = cv.waitKey(1) & 0xFF
        
        # Press SPACE to capture
        if key == ord(' '):
            print("Image captured!")
            cap.release()
            cv.destroyAllWindows()
            return frame
        
        # Press ESC to exit
        elif key == 27:
            print("Capture cancelled.")
            cap.release()
            cv.destroyAllWindows()
            return None
    
    cap.release()
    cv.destroyAllWindows()
    return None


def load_from_file():
    """
    Load an image from a file path provided by user.
    
    Returns:
        Tuple of (image, image_name), or (None, None) if loading fails
    """
    while True:
        file_path = input("\nEnter the path to the card image: ").strip()
        
        # Remove quotes if user wrapped the path
        file_path = file_path.strip('"').strip("'")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None, None
            continue
        
        image = cv.imread(file_path)
        
        if image is None:
            print(f"Error: Could not load image from '{file_path}'")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None, None
            continue
        
        image_name = os.path.basename(file_path)
        print(f"Successfully loaded: {image_name}")
        return image, image_name
    
    return None, None


def process_card(image: np.ndarray, image_name: str = "card") -> None:
    """
    Complete pipeline: rotate, perspective transform, detect and classify card.
    
    Args:
        image: Input card image
        image_name: Name for the image (for display purposes)
    """
    print("\n" + "-"*50)
    print("Processing card...")
    print("-"*50)
    
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Step 1: Rotate and perspective transform
    print("\n[1/3] Applying rotation and perspective transformation...")
    rotator = CardRotator(image_path="")  # Not using file path for direct processing
    
    # Threshold to find contours
    thresholded = rotator.threshold(gray_image)
    largest_contour, _ = rotator.find_largest_contour(thresholded)
    
    # Apply perspective transformation to get a flat, front-facing card
    perspective_card = rotator.apply_perspective_transform(thresholded, largest_contour)
    
    # Crop to remove excess background
    mask = rotator.create_mask(perspective_card)
    cropped_card = rotator.crop_image(perspective_card, mask)
    
    print(f"   ✓ Card rotated and flattened (size: {cropped_card.shape})")
    
    # Step 2: Detect and classify the card
    print("\n[2/3] Detecting and classifying card...")
    
    # Initialize detector with training images directories
    ranks_path = "./RANKS/"
    suits_path = "./SUITS/"
    
    # Check if directories exist
    if not os.path.exists(ranks_path):
        print(f"Error: Ranks directory not found at '{ranks_path}'")
        print("Please create a RANKS/ folder with rank template images.")
        return
    
    if not os.path.exists(suits_path):
        print(f"Error: Suits directory not found at '{suits_path}'")
        print("Please create a SUITS/ folder with suit template images.")
        return
    
    detector = CardDetector(ranks_path=ranks_path, suits_path=suits_path)
    
    # Classify the card
    card_identity = detector.detect_and_classify(cropped_card)
    
    print(f"   ✓ Detection complete")
    
    # Step 3: Display results
    print("\n[3/3] Results:")
    print("="*50)
    
    if card_identity.is_valid():
        print(f"   Card Identified: {card_identity}")
        print(f"   Confidence: {card_identity.get_confidence_info()}")
    else:
        print(f"   Card: {card_identity}")
        print("   ⚠ Card could not be fully identified")
        print(f"   Details: {card_identity.get_confidence_info()}")
    
    print("="*50)
    
    # Display the processed card
    cv.imshow(f'Original Image - {image_name}', image)
    cv.imshow('Processed Card (Rotated & Perspective Corrected)', cropped_card)
    
    # Display rank and suit if available
    if card_identity.rank_img is not None:
        cv.imshow('Detected Rank', card_identity.rank_img)
    if card_identity.suit_img is not None:
        cv.imshow('Detected Suit', card_identity.suit_img)
    
    # Display difference maps (delta maps) for matching visualization
    if card_identity.rank_diff_img is not None:
        # Apply colormap for better visualization
        rank_diff_colored = cv.applyColorMap(card_identity.rank_diff_img, cv.COLORMAP_JET)
        cv.imshow('Rank Difference Map (Best Match)', rank_diff_colored)
    
    if card_identity.suit_diff_img is not None:
        suit_diff_colored = cv.applyColorMap(card_identity.suit_diff_img, cv.COLORMAP_JET)
        cv.imshow('Suit Difference Map (Best Match)', suit_diff_colored)
    
    print("\nPress any key to close windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Ask if user wants to see intermediate steps
    viz_choice = input("\nWould you like to see the intermediate processing steps? (y/n): ").strip().lower()
    if viz_choice == 'y':
        detector.visualize_intermediate_steps(card_identity)
    
    # Ask user if they want to save the detected rank and suit
    if card_identity.rank_img is not None and card_identity.suit_img is not None:
        save_choice = input("\nWould you like to save the detected rank and suit images? (y/n): ").strip().lower()
        
        if save_choice == 'y':
            save_detected_images(card_identity, image_name)


def save_detected_images(card_identity: CardIdentity, image_name: str) -> None:
    """
    Save the detected rank and suit images to their respective folders.
    If unknown, prompt user for the correct rank/suit.
    
    Args:
        card_identity: CardIdentity object containing the detected images
        image_name: Original image name (used for generating unique filenames)
    """
    rank_to_save = card_identity.rank
    suit_to_save = card_identity.suit
    
    # If rank is unknown, ask user for it
    if card_identity.rank == "Unknown":
        print("\n⚠ Rank was not detected.")
        rank_to_save = input("Please enter the rank (e.g., Ace, Two, Three, ..., King): ").strip().capitalize()
        if not rank_to_save:
            print("   ✗ Skipping rank save - no input provided")
            rank_to_save = None
    
    # If suit is unknown, ask user for it
    if card_identity.suit == "Unknown":
        print("\n⚠ Suit was not detected.")
        suit_to_save = input("Please enter the suit (Spades, Diamonds, Clubs, Hearts): ").strip().capitalize()
        if not suit_to_save:
            print("   ✗ Skipping suit save - no input provided")
            suit_to_save = None
    
    # Save rank image
    if card_identity.rank_img is not None and rank_to_save:
        rank_folder = "./RANKS/"
        os.makedirs(rank_folder, exist_ok=True)
        
        # Find next available number for this rank
        rank_number = get_next_number(rank_folder, rank_to_save)
        rank_filename = os.path.join(rank_folder, f"{rank_to_save}{rank_number}.jpg")
        
        cv.imwrite(rank_filename, card_identity.rank_img)
        print(f"   ✓ Saved rank image: {rank_filename}")
    
    # Save suit image
    if card_identity.suit_img is not None and suit_to_save:
        suit_folder = "./SUITS/"
        os.makedirs(suit_folder, exist_ok=True)
        
        # Find next available number for this suit
        suit_number = get_next_number(suit_folder, suit_to_save)
        suit_filename = os.path.join(suit_folder, f"{suit_to_save}{suit_number}.jpg")
        
        cv.imwrite(suit_filename, card_identity.suit_img)
        print(f"   ✓ Saved suit image: {suit_filename}")


def get_next_number(folder: str, prefix: str) -> int:
    """
    Find the next available number for a given prefix in a folder.
    
    Args:
        folder: Directory to search in
        prefix: Filename prefix (e.g., "Ace", "Spades")
    
    Returns:
        Next available number (0-9, cycling back to 0 if all are taken)
    """
    existing_files = []
    if os.path.exists(folder):
        existing_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.jpg')]
    
    # Extract numbers from existing files
    used_numbers = set()
    for filename in existing_files:
        # Remove prefix and .jpg extension
        num_str = filename[len(prefix):-4]
        if num_str.isdigit():
            used_numbers.add(int(num_str))
    
    # Find first available number 0-9
    for i in range(10):
        if i not in used_numbers:
            return i
    
    # If all 0-9 are taken, start overwriting from 0
    return 0 


def main():
    """Main application entry point."""
    
    # Get user's choice
    choice = get_user_choice()
    
    # Get the image
    if choice == 'webcam':
        image = capture_from_webcam()
        image_name = "webcam_capture"
    else:  # file
        image, image_name = load_from_file()
    
    # Check if we got a valid image
    if image is None:
        print("\nNo image to process. Exiting.")
        return
    
    # Ensure image_name is set
    if image_name is None:
        image_name = "unknown"
    
    # Process the card
    try:
        process_card(image, image_name)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nThank you for using the Card Detection System!")


if __name__ == "__main__":
    main()
