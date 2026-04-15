"""Utility class for floor edge detection in robot vision.

Provides image processing pipeline for detecting floor boundaries
using color segmentation and contour analysis on downward-facing images.
"""
import cv2
import numpy as np


class Floor:
    def __init__(self):
        """
        Initialize the Floor class.
        """
        # Processing parameters
        self.blur_kernel_size = 0
        self.crop_height = 0
        self.crop_width = 0
        self.resize_width = 0
        self.resize_height = 0
        self.bottom_sample_width_fraction = 0
        self.bottom_sample_height_fraction = 0
        self.grad_thresh = 0
        self.color_thresh = 0
        self.morph_kernel_size = 0
        self.val_range = 0
        self.highlight_thresh = 0

        # Output data
        self.floor_mask = None
        self.blurred = None
        self.grad_mag = None
        self.color_dist = None
        self.base_floor = None
        self.reflection_mask = None

        # Output enable flags (set to True to save intermediate results)
        self.save_floor_mask = False
        self.save_blurred = False
        self.save_grad_mag = False
        self.save_color_dist = False
        self.save_base_floor = False
        self.save_reflection_mask = False


    # -------------------------
    # Setters
    # -------------------------

    def set_image_crop_size(self, height, width):
        """
        Set the image crop size for processing.

        :param height: Desired height for cropping (integer)
        :param width: Desired width for cropping (integer)

        Cropped image will be the bottom center of the original image with the specified height and width.
        """
        self.crop_height = height
        self.crop_width = width

    def set_resize_dimensions(self, width, height):
        """
        Set the dimensions for resizing the image.

        :param width: Desired width for resizing (integer)
        :param height: Desired height for resizing (integer)
        """
        self.resize_width = width
        self.resize_height = height

    def set_blur_kernel_size(self, k):
        """
        Set the blur kernel size, ensuring it is a positive odd integer.

        :param k: Desired kernel size (integer)
        """
        if k % 2 == 0:
            k += 1
        self.blur_kernel_size = max(3, k)

    def set_bottom_sample_size(self, width_fraction, height_fraction):
        """
        Set the size of the bottom sampling region for floor color statistics.

        :param width_fraction: Fraction of the image width to sample (0.0 to 1.0)
        :param height_fraction: Fraction of the image height to sample (0.0 to 1.0)

        The sampling region will be centered horizontally at the bottom of the image.
        """
        self.bottom_sample_width_fraction = width_fraction
        self.bottom_sample_height_fraction = height_fraction

    def set_gradient_threshold(self, thresh):
        """Set the edge gradient magnitude threshold for edge detection."""
        self.grad_thresh = thresh

    def set_color_distance_threshold(self, thresh):
        """Set the color distance threshold for floor color segmentation."""
        self.color_thresh = thresh

    def set_morph_kernel_size(self, size):
        """Set the kernel size for morphological operations to clean up the floor mask."""
        self.morph_kernel_size = size

    def set_val_range(self, val_range):
        """Set the brightness (L channel) range for floor pixel acceptance."""
        self.val_range = val_range

    def set_highlight_thresh(self, thresh):
        """Set the highlight threshold above which pixels are considered specular highlights."""
        self.highlight_thresh = thresh


    # -------------------------
    # Output Enable Setters
    # -------------------------

    def enable_floor_mask(self, enable=True):
        """Enable or disable saving the floor mask output."""
        self.save_floor_mask = enable

    def enable_blurred(self, enable=True):
        """Enable or disable saving the blurred (and resized) image output."""
        self.save_blurred = enable

    def enable_grad_mag(self, enable=True):
        """Enable or disable saving the gradient magnitude output."""
        self.save_grad_mag = enable

    def enable_color_dist(self, enable=True):
        """Enable or disable saving the color distance output."""
        self.save_color_dist = enable

    def enable_base_floor(self, enable=True):
        """Enable or disable saving the base floor mask output."""
        self.save_base_floor = enable

    def enable_reflection_mask(self, enable=True):
        """Enable or disable saving the reflection mask output."""
        self.save_reflection_mask = enable


    # -------------------------
    # Parameter Getters
    # -------------------------

    def get_blur_kernel_size(self):
        """Return the current blur kernel size."""
        return self.blur_kernel_size

    def get_crop_size(self):
        """Return the crop dimensions as (height, width)."""
        return (self.crop_height, self.crop_width)

    def get_resize_dimensions(self):
        """Return the resize dimensions as (width, height)."""
        return (self.resize_width, self.resize_height)

    def get_bottom_sample_size(self):
        """Return the bottom sample fractions as (width_fraction, height_fraction)."""
        return (self.bottom_sample_width_fraction, self.bottom_sample_height_fraction)

    def get_gradient_threshold(self):
        """Return the gradient threshold value."""
        return self.grad_thresh

    def get_color_distance_threshold(self):
        """Return the color distance threshold value."""
        return self.color_thresh

    def get_morph_kernel_size(self):
        """Return the morphological kernel size."""
        return self.morph_kernel_size

    def get_val_range(self):
        """Return the brightness value range threshold."""
        return self.val_range

    def get_highlight_thresh(self):
        """Return the highlight threshold value."""
        return self.highlight_thresh


    # -------------------------
    # Result Getters
    # -------------------------

    def get_floor_mask(self):
        """Return the computed floor mask."""
        return self.floor_mask

    def get_blurred(self):
        """Return the blurred (and optionally resized) image used for processing."""
        return self.blurred

    def get_grad_mag(self):
        """Return the gradient magnitude image."""
        return self.grad_mag

    def get_color_dist(self):
        """Return the color distance image."""
        return self.color_dist

    def get_base_floor(self):
        """Return the base floor mask (prior to merging with the reflection mask)."""
        return self.base_floor

    def get_reflection_mask(self):
        """Return the reflection mask."""
        return self.reflection_mask


    # -------------------------
    # Main Processing
    # -------------------------

    def find_floor(self, frame):
        """
        Detect the floor region in the given image frame and store results
        in instance variables accessible via get_* methods.

        Only operations whose controlling parameter is non-zero are executed.
        Only outputs whose enable flag is True are saved to self.*

        :param frame: Input OpenCV BGR image
        """
        # Resize first if resize dimensions are set
        if self.resize_width > 0 and self.resize_height > 0:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))

        # Apply Gaussian blur if kernel size is set
        if self.blur_kernel_size > 2:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # Crop to bottom-center region if crop dimensions are set
        if self.crop_height > 0 and self.crop_width > 0:
            h_cur, w_cur = frame.shape[:2]
            if h_cur > self.crop_height and w_cur > self.crop_width:
                x0 = w_cur // 2 - self.crop_width // 2
                frame = frame[h_cur - self.crop_height:h_cur, x0:x0 + self.crop_width]

        if self.save_blurred:
            self.blurred = frame

        # Extract LAB color channels for better color segmentation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        h, w = L.shape

        # Select image sampling region for floor color statistics
        if self.bottom_sample_width_fraction > 0 and self.bottom_sample_height_fraction > 0:
            sample_start_y = int((1 - self.bottom_sample_height_fraction) * h)
            sample_start_x = int(w * (1 - self.bottom_sample_width_fraction) / 2)
            sample_end_x = int(w * self.bottom_sample_width_fraction + sample_start_x)
        else:
            sample_start_y = 0
            sample_start_x = 0
            sample_end_x = w

        sample_L = L[sample_start_y:, sample_start_x:sample_end_x]
        sample_A = A[sample_start_y:, sample_start_x:sample_end_x]
        sample_B = B[sample_start_y:, sample_start_x:sample_end_x]

        mean_L = np.mean(sample_L)
        mean_A = np.mean(sample_A)
        mean_B = np.mean(sample_B)

        val_mask = np.abs(L - mean_L) < self.val_range if self.val_range > 0 else np.ones(L.shape, dtype=bool)

        # Calculate color distance from the sampled floor color
        color_dist = np.sqrt((A - mean_A)**2 + (B - mean_B)**2)
        if self.save_color_dist:
            self.color_dist = color_dist

        # Compute gradient magnitude for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad = cv2.Laplacian(gray, cv2.CV_64F)
        grad_mag = np.abs(grad)
        if self.save_grad_mag:
            self.grad_mag = grad_mag
        edge_mask = grad_mag < self.grad_thresh if self.grad_thresh > 0 else np.ones(L.shape, dtype=bool)

        # Define a reflection mask: bright pixels with similar color and low edge magnitude
        if self.color_thresh > 0 and self.grad_thresh > 0:
            reflection_mask = (
                (L > mean_L) &
                (color_dist < self.color_thresh) &
                (grad_mag < self.grad_thresh)
            )
        else:
            reflection_mask = np.zeros(L.shape, dtype=bool)
        if self.save_reflection_mask:
            self.reflection_mask = reflection_mask.astype(np.uint8) * 255

        # Define the base floor mask using brightness, color similarity, and edge magnitude
        if self.color_thresh > 0:
            base_floor = (
                val_mask &
                (color_dist < self.color_thresh) &
                edge_mask
            )
        else:
            base_floor = val_mask & edge_mask
        if self.save_base_floor:
            self.base_floor = base_floor.astype(np.uint8) * 255

        combined_mask = (base_floor | reflection_mask)
        floor_mask = combined_mask.astype(np.uint8) * 255

        # Apply morphological operations if kernel size is set
        if self.morph_kernel_size > 0:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

        # Flood fill from bottom-center to retain only floor connected to bottom of image
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        ff = floor_mask.copy()
        cv2.floodFill(ff, flood_mask, (w // 2, h - 1), 255)
        floor_mask = ff

        # Hole filling: inverse flood fill to detect and fill enclosed holes
        inv = cv2.bitwise_not(floor_mask)
        flood_mask2 = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inv, flood_mask2, (0, 0), 255)
        holes = cv2.bitwise_not(inv)
        if self.save_floor_mask:
            self.floor_mask = cv2.bitwise_or(floor_mask, holes)




