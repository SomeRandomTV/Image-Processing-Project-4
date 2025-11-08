import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class CardRotator:
    def __init__(self, image_path: str):
        """
        Initializes the CardDeskewer class with the path to the directory containing card images.
        """
        self.image_path = image_path

    # ---------------- Utility Methods ----------------

    @staticmethod
    def threshold(image: np.ndarray, threshold_type: int = cv.THRESH_OTSU) -> np.ndarray:
        _, thresholded_image = cv.threshold(image, 0, 255, threshold_type)
        return thresholded_image

    @staticmethod
    def find_largest_contour(thresholded_image: np.ndarray) -> tuple:
        contours, _ = cv.findContours(
            thresholded_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv.contourArea)
        return largest_contour, contours

    @staticmethod
    def draw_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2 or image.shape[2] == 1:
            image_copy = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        else:
            image_copy = image.copy()
        cv.drawContours(image_copy, [contour], -1, (0, 255, 0), 5)
        return image_copy

    @staticmethod
    def calculate_orientation(largest_contour: np.ndarray) -> int:
        data = largest_contour.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv.PCACompute(data, mean=None)[:2]
        angle = int(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi)
        return angle

    @staticmethod
    def rotate_to_vertical_affine(image: np.ndarray, orientation: int) -> np.ndarray:
        orientation = -orientation % 180
        angle_to_rotate = int(90 - orientation)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle_to_rotate, 1.0)
        result = cv.warpAffine(image, M, (w, h))
        return result.astype(np.uint8)

    @staticmethod
    def rotate_to_vertical_projective(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect).astype(np.float32)

        s = box.sum(axis=1)
        tl = box[np.argmin(s)]
        br = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        tr = box[np.argmin(diff)]
        bl = box[np.argmax(diff)]
        ordered = np.array([tl, tr, br, bl], dtype="float32")

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32",
        )

        M = cv.getPerspectiveTransform(ordered, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def create_mask(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
        mask = (image > threshold_value).astype(np.uint8) * 255
        return mask

    @staticmethod
    def crop_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        coords = cv.findNonZero(mask)
        if coords is None:
            return image
        x, y, w, h = cv.boundingRect(coords)
        return image[y:y + h, x:x + w]

    @staticmethod
    def apply_perspective_transform(image: np.ndarray, contour: np.ndarray, 
                                    target_width: int = 200, target_height: int = 300) -> np.ndarray:
        """
        Apply perspective transformation to make the card front and center.
        This corrects any perspective distortion from camera angle.
        
        Args:
            image: Input image containing the card
            contour: Contour of the card
            target_width: Desired width of the output card (default: 200)
            target_height: Desired height of the output card (default: 300)
            
        Returns:
            np.ndarray: Perspective-corrected card image (always vertical)
        """
        # Get the minimum area rectangle around the contour
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect).astype(np.float32)
        
        # Order the points: top-left, top-right, bottom-right, bottom-left
        # This ordering is critical for correct perspective transform
        def order_points(pts):
            # Initialize ordered coordinates
            rect = np.zeros((4, 2), dtype="float32")
            
            # Top-left has smallest sum, bottom-right has largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            
            # Top-right has smallest difference, bottom-left has largest difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            
            return rect
        
        ordered = order_points(box)
        
        # Calculate width and height of the bounding box
        (tl, tr, br, bl) = ordered
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Determine if card needs rotation (if it's wider than tall)
        if maxWidth > maxHeight:
            # Card is horizontal, we need to swap the corner order
            # to make it vertical after transform
            final_width = target_width
            final_height = target_height
            
            # Rotate the point order 90 degrees: [tl, tr, br, bl] -> [bl, tl, tr, br]
            ordered = np.array([bl, tl, tr, br], dtype="float32")
        else:
            # Card is already vertical
            final_width = target_width
            final_height = target_height
        
        # Define destination points for perspective transform
        # Always map to a vertical card
        dst = np.array([
            [0, 0],                              # top-left
            [final_width - 1, 0],                # top-right
            [final_width - 1, final_height - 1], # bottom-right
            [0, final_height - 1]                # bottom-left
        ], dtype="float32")
        
        # Calculate and apply perspective transform
        M = cv.getPerspectiveTransform(ordered, dst)
        warped = cv.warpPerspective(image, M, (final_width, final_height))
        
        return warped

    @staticmethod
    def show_image(
        gray_image: np.ndarray, result: np.ndarray = None, title: str = None, **kwargs
    ):
        if result is None:
            print("result is None in show_image, skipping display.")
            return

        orientation = kwargs.get("orientation", None)
        final_orientation = kwargs.get("final_orientation", None)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle(title)
        ax_img_orig, ax_hist_orig = axes[0]
        ax_img_thr, ax_hist_thr = axes[1]

        hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
        ax_img_orig.imshow(gray_image, cmap="gray")
        ax_img_orig.set_title(f"(Original), Orientation: {orientation}")
        ax_img_orig.axis("off")
        ax_hist_orig.plot(hist, color="black")
        ax_hist_orig.set_title("Histogram (Original)")
        ax_hist_orig.grid(True)

        hist_result = cv.calcHist([result], [0], None, [256], [0, 256])
        ax_img_thr.imshow(result, cmap="gray")
        ax_img_thr.set_title(f"(Final), Orientation: {final_orientation}")
        ax_img_thr.axis("off")
        ax_hist_thr.plot(hist_result, color="black")
        ax_hist_thr.set_title("Histogram (Final)")
        ax_hist_thr.grid(True)

        plt.show()

    # ---------------- Core Processing ----------------

    def process_image(self, imname: str, counter: int):
        print(f"{'='*15} Processing Testimage{counter} ({imname}) {'='*15}\n")

        image = cv.imread(os.path.join(self.image_path, imname), cv.IMREAD_GRAYSCALE)
        thresholded_image = self.threshold(image)
        largest_contour, _ = self.find_largest_contour(thresholded_image)
        initial_orientation = self.calculate_orientation(largest_contour)

        # Apply perspective transformation to get front and center view
        perspective_card = self.apply_perspective_transform(image, largest_contour)
        
        # Get new contour after perspective correction
        thresh_perspective = self.threshold(perspective_card)
        corrected_contour, _ = self.find_largest_contour(thresh_perspective)
        corrected_orientation = self.calculate_orientation(corrected_contour)

        # Crop
        mask = self.create_mask(perspective_card)
        cropped_card = self.crop_image(perspective_card, mask)

        # Visualization
        print(f"Largest initial contour area: {cv.contourArea(largest_contour)}")
        print(f"Initial orientation: {initial_orientation}")
        print(f"Corrected orientation: {corrected_orientation}")
        print(f"{'-'*10}")

        self.show_image(
            image,
            cropped_card,
            orientation=initial_orientation,
            final_orientation=corrected_orientation,
            title="Perspective Correction"
        )
        return cropped_card


