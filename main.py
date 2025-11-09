import cv2 as cv
import numpy as np
import os
from card_rotator import CardRotator
from card_detector import CardDetector, CardIdentity
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


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
    print("  CARD DETECTION RESULTS")
    print("="*50)
    print()
    
    if card_identity.is_valid():
        print(f"Card Identified:")
        print(f"  {card_identity}")
        print()
        print("Status: SUCCESS")
    else:
        print(f"Card: {card_identity}")
        print()
        print("Status: PARTIAL/UNKNOWN")
    
    print()
    print("-"*50)
    print("Confidence Scores:")
    print("-"*50)
    print()
    
    # Calculate match quality percentages
    rank_quality = max(0, (1 - card_identity.rank_confidence / 2000) * 100) if card_identity.rank != "Unknown" else 0
    suit_quality = max(0, (1 - card_identity.suit_confidence / 700) * 100) if card_identity.suit != "Unknown" else 0
    overall_confidence = (rank_quality + suit_quality) / 2 if card_identity.is_valid() else 0
    
    print("Rank:")
    print(f"  Name: {card_identity.rank}")
    print(f"  Difference: {card_identity.rank_confidence}")
    print(f"  Threshold: 2000")
    print(f"  Match Quality: {rank_quality:.2f}%")
    print(f"  Status: {'MATCHED' if card_identity.rank != 'Unknown' else 'UNMATCHED'}")
    print()
    
    print("Suit:")
    print(f"  Name: {card_identity.suit}")
    print(f"  Difference: {card_identity.suit_confidence}")
    print(f"  Threshold: 700")
    print(f"  Match Quality: {suit_quality:.2f}%")
    print(f"  Status: {'MATCHED' if card_identity.suit != 'Unknown' else 'UNMATCHED'}")
    print()
    
    print("-"*50)
    print(f"Overall Confidence: {overall_confidence:.2f}%")
    print("-"*50)
    print()
    print("Note: Lower difference = Better match")
    print("      Match Quality = (1 - diff/threshold) * 100%")
    
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


class CardDetectionGUI:
    """GUI application for card detection."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Card Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        self.current_image = None
        self.current_image_name = None
        self.card_identity = None
        self.detector = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Card Detection System",
            font=("Arial", 24, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        title_label.pack(pady=20)
        
        # Control Frame
        control_frame = tk.Frame(self.root, bg='#2b2b2b')
        control_frame.pack(pady=10)
        
        # Buttons
        btn_style = {
            'font': ('Arial', 12),
            'width': 20,
            'height': 2,
            'bg': '#4a9eff',
            'fg': 'black',
            'relief': 'raised',
            'borderwidth': 2
        }
        
        self.webcam_btn = tk.Button(
            control_frame,
            text="Capture from Webcam",
            command=self.capture_webcam,
            **btn_style
        )
        self.webcam_btn.grid(row=0, column=0, padx=10)
        
        self.file_btn = tk.Button(
            control_frame,
            text="Load from File",
            command=self.load_file,
            **btn_style
        )
        self.file_btn.grid(row=0, column=1, padx=10)
        
        self.process_btn = tk.Button(
            control_frame,
            text="Process Card",
            command=self.process_current_card,
            state='disabled',
            **btn_style
        )
        self.process_btn.grid(row=0, column=2, padx=10)
        
        # Results Frame
        results_frame = tk.Frame(self.root, bg='#3b3b3b', relief='groove', borderwidth=3)
        results_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Image Display Area
        image_frame = tk.Frame(results_frame, bg='#3b3b3b')
        image_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)
        
        # Original Image
        tk.Label(
            image_frame, 
            text="Original Image",
            font=("Arial", 14, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        ).pack()
        
        self.original_canvas = tk.Canvas(
            image_frame,
            width=400,
            height=300,
            bg='#1b1b1b',
            highlightthickness=0
        )
        self.original_canvas.pack(pady=5)
        
        # Processed Image
        tk.Label(
            image_frame, 
            text="Processed Card",
            font=("Arial", 14, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        ).pack(pady=(10, 0))
        
        self.processed_canvas = tk.Canvas(
            image_frame,
            width=400,
            height=300,
            bg='#1b1b1b',
            highlightthickness=0
        )
        self.processed_canvas.pack(pady=5)
        
        # Results Display Area
        info_frame = tk.Frame(results_frame, bg='#3b3b3b')
        info_frame.pack(side='right', padx=10, pady=10, fill='both', expand=True)
        
        # Card Identity Display
        tk.Label(
            info_frame, 
            text="Detection Results",
            font=("Arial", 16, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        ).pack(pady=(0, 20))
        
        self.result_text = tk.Text(
            info_frame,
            width=40,
            height=15,
            font=("Courier", 12),
            bg='#1b1b1b',
            fg='#00ff00',
            relief='sunken',
            borderwidth=2,
            state='disabled'
        )
        self.result_text.pack(pady=10)
        
        # Extracted Symbols Frame
        symbols_frame = tk.Frame(info_frame, bg='#3b3b3b')
        symbols_frame.pack(pady=10)
        
        # Rank Display
        rank_frame = tk.Frame(symbols_frame, bg='#3b3b3b')
        rank_frame.pack(side='left', padx=10)
        
        tk.Label(
            rank_frame, 
            text="Detected Rank",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        ).pack()
        
        self.rank_canvas = tk.Canvas(
            rank_frame,
            width=100,
            height=150,
            bg='#1b1b1b',
            highlightthickness=0
        )
        self.rank_canvas.pack(pady=5)
        
        # Suit Display
        suit_frame = tk.Frame(symbols_frame, bg='#3b3b3b')
        suit_frame.pack(side='left', padx=10)
        
        tk.Label(
            suit_frame, 
            text="Detected Suit",
            font=("Arial", 11, "bold"),
            bg='#3b3b3b',
            fg='#ffffff'
        ).pack()
        
        self.suit_canvas = tk.Canvas(
            suit_frame,
            width=100,
            height=150,
            bg='#1b1b1b',
            highlightthickness=0
        )
        self.suit_canvas.pack(pady=5)
        
        # Status Bar
        self.status_label = tk.Label(
            self.root,
            text="Ready. Please capture from webcam or load an image file.",
            font=("Arial", 10),
            bg='#1b1b1b',
            fg='#00ff00',
            anchor='w',
            relief='sunken',
            borderwidth=1
        )
        self.status_label.pack(side='bottom', fill='x')
    
    def update_status(self, message, color='#00ff00'):
        """Update status bar message."""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def capture_webcam(self):
        """Capture image from webcam."""
        self.update_status("Opening webcam...", '#ffaa00')
        
        cap = cv.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.update_status("Error: Could not open webcam.", '#ff0000')
            return
        
        # Create a simple capture window
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Webcam Capture")
        capture_window.geometry("640x520")
        
        instruction = tk.Label(
            capture_window,
            text="Press SPACE to capture, ESC to cancel",
            font=("Arial", 12, "bold")
        )
        instruction.pack(pady=10)
        
        video_label = tk.Label(capture_window)
        video_label.pack()
        
        captured_image = [None]  # Use list to store result
        
        def show_frame():
            ret, frame = cap.read()
            if ret:
                # Convert for display
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
            
            if captured_image[0] is None:
                video_label.after(10, show_frame)
        
        def on_key(event):
            if event.char == ' ':  # Space pressed
                ret, frame = cap.read()
                if ret:
                    captured_image[0] = frame
                    cap.release()
                    capture_window.destroy()
            elif event.keycode == 27:  # ESC pressed
                cap.release()
                capture_window.destroy()
        
        capture_window.bind('<Key>', on_key)
        show_frame()
        
        capture_window.wait_window()
        
        if captured_image[0] is not None:
            self.current_image = captured_image[0]
            self.current_image_name = "webcam_capture"
            self.display_original_image(self.current_image)
            self.process_btn.config(state='normal')
            self.update_status(f"Image captured: {self.current_image_name}")
        else:
            self.update_status("Capture cancelled.", '#ffaa00')
    
    def load_file(self):
        """Load image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Card Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.update_status(f"Loading image from {file_path}...", '#ffaa00')
        
        image = cv.imread(file_path)
        
        if image is None:
            messagebox.showerror("Error", f"Could not load image from '{file_path}'")
            self.update_status("Error: Could not load image.", '#ff0000')
            return
        
        self.current_image = image
        self.current_image_name = os.path.basename(file_path)
        self.display_original_image(image)
        self.process_btn.config(state='normal')
        self.update_status(f"Image loaded: {self.current_image_name}")
    
    def display_original_image(self, image):
        """Display original image on canvas."""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        
        # Resize to fit canvas
        img = Image.fromarray(image_rgb)
        img.thumbnail((400, 300), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.original_canvas.delete("all")
        self.original_canvas.create_image(200, 150, image=photo)
        self.original_canvas.image = photo
    
    def display_processed_image(self, image):
        """Display processed card image on canvas."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to fit canvas
        img = Image.fromarray(image_rgb)
        img.thumbnail((400, 300), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img)
        self.processed_canvas.delete("all")
        self.processed_canvas.create_image(200, 150, image=photo)
        self.processed_canvas.image = photo
    
    def display_symbol(self, canvas, image):
        """Display rank or suit symbol on canvas."""
        if image is None:
            canvas.delete("all")
            canvas.create_text(50, 75, text="N/A", fill='white', font=("Arial", 12))
            return
        
        # Convert to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        img = Image.fromarray(image_rgb)
        
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(50, 75, image=photo)
        canvas.image = photo
    
    def update_results_text(self, card_identity):
        """Update results text display."""
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        # Calculate match quality percentages
        rank_quality = max(0, (1 - card_identity.rank_confidence / 2000) * 100) if card_identity.rank != "Unknown" else 0
        suit_quality = max(0, (1 - card_identity.suit_confidence / 700) * 100) if card_identity.suit != "Unknown" else 0
        overall_confidence = (rank_quality + suit_quality) / 2 if card_identity.is_valid() else 0
        
        result = "="*40 + "\n"
        result += "  CARD DETECTION RESULTS\n"
        result += "="*40 + "\n\n"
        
        if card_identity.is_valid():
            result += f"Card Identified:\n"
            result += f"  {card_identity}\n\n"
            result += f"Status: SUCCESS\n\n"
        else:
            result += f"Card: {card_identity}\n\n"
            result += f"Status: PARTIAL/UNKNOWN\n\n"
        
        result += "-"*40 + "\n"
        result += "Confidence Scores:\n"
        result += "-"*40 + "\n\n"
        
        result += f"Rank:\n"
        result += f"  Name: {card_identity.rank}\n"
        result += f"  Difference: {card_identity.rank_confidence}\n"
        result += f"  Threshold: 2000\n"
        result += f"  Match Quality: {rank_quality:.2f}%\n"
        
        if card_identity.rank != "Unknown":
            result += f"  Status: MATCHED\n\n"
        else:
            result += f"  Status: UNMATCHED\n\n"
        
        result += f"Suit:\n"
        result += f"  Name: {card_identity.suit}\n"
        result += f"  Difference: {card_identity.suit_confidence}\n"
        result += f"  Threshold: 700\n"
        result += f"  Match Quality: {suit_quality:.2f}%\n"
        
        if card_identity.suit != "Unknown":
            result += f"  Status: MATCHED\n\n"
        else:
            result += f"  Status: UNMATCHED\n\n"
        
        result += "-"*40 + "\n"
        result += f"Overall Confidence: {overall_confidence:.2f}%\n"
        result += "-"*40 + "\n\n"
        
        result += "Note: Lower difference = Better match\n"
        result += "      Match Quality = (1 - diff/threshold) * 100%\n"
        
        self.result_text.insert(1.0, result)
        self.result_text.config(state='disabled')
    
    def process_current_card(self):
        """Process the current loaded card."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded. Please capture or load an image first.")
            return
        
        self.update_status("Processing card...", '#ffaa00')
        
        try:
            # Convert to grayscale
            if len(self.current_image.shape) == 3:
                gray_image = cv.cvtColor(self.current_image, cv.COLOR_BGR2GRAY)
            else:
                gray_image = self.current_image
            
            # Step 1: Rotate and perspective transform
            rotator = CardRotator(image_path="")
            thresholded = rotator.threshold(gray_image)
            largest_contour, _ = rotator.find_largest_contour(thresholded)
            perspective_card = rotator.apply_perspective_transform(thresholded, largest_contour)
            mask = rotator.create_mask(perspective_card)
            cropped_card = rotator.crop_image(perspective_card, mask)
            
            # Display processed card
            self.display_processed_image(cropped_card)
            
            # Step 2: Detect and classify
            ranks_path = "./RANKS/"
            suits_path = "./SUITS/"
            
            if not os.path.exists(ranks_path) or not os.path.exists(suits_path):
                messagebox.showerror(
                    "Error",
                    "RANKS or SUITS directory not found.\nPlease create template directories."
                )
                self.update_status("Error: Template directories not found.", '#ff0000')
                return
            
            # Initialize detector only once
            if self.detector is None:
                self.detector = CardDetector(ranks_path=ranks_path, suits_path=suits_path)
            
            # Classify
            self.card_identity = self.detector.detect_and_classify(cropped_card)
            
            # Update displays
            self.display_symbol(self.rank_canvas, self.card_identity.rank_img)
            self.display_symbol(self.suit_canvas, self.card_identity.suit_img)
            self.update_results_text(self.card_identity)
            
            if self.card_identity.is_valid():
                self.update_status(f"Success! Detected: {self.card_identity}", '#00ff00')
            else:
                self.update_status(f"Partial detection: {self.card_identity}", '#ffaa00')
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing:\n{str(e)}")
            self.update_status(f"Error: {str(e)}", '#ff0000')
            import traceback
            traceback.print_exc()


def main():
    """Main application entry point."""
    root = tk.Tk()
    app = CardDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
