from python.sift_extractor import SIFT
import numpy as np
import cv2
from typing import List
from sklearn.neighbors import NearestNeighbors
from python.general_functions import show_images, normalize_img
from skimage.color import rgb2gray


class Panorama:
    def __init__(self):
        pass

    def compute_homography_matrix(self, src_sift, dst_sift, nearest_neighbors_model, threshold_distance=0.7):
        '''calculates homography matrix of transforming source image to destination image'''
        
        # Find nearest neighbors for each descriptor in the source keypoints
        distances, indices = nearest_neighbors_model.kneighbors(src_sift["Descriptor"].to_list())

        # Filter indices based on the threshold distance
        filtered_indices = indices[distances < threshold_distance].ravel()

        # Extract corresponding points from the source and target keypoints
        source_points = np.array(src_sift.iloc[distances < threshold_distance]['Position'].to_list())[:, :2]
        target_points = np.array(dst_sift.take(filtered_indices)['Position'].to_list())[:, :2]

        # Estimate homography matrix using RANSAC
        # computes the homography matrix (H) to transform points from the source image (srcPoints) to the destination image (dstPoints)
        homography_matrix, _ = cv2.findHomography(source_points.astype(np.float32), target_points.astype(np.float32), 
                                                cv2.RANSAC, ransacReprojThreshold=5)

        return homography_matrix

    def stitch_two_images(self, img1, img2, x_offset=0, img1_left=False):
        """
        Stitch two images horizontally.

        Parameters:
        - img1: First input image to be stitched.
        - img2: Second input image to be stitched.
        - x_offset: Horizontal offset to adjust the stitching position of img1.
        - img1_left: Boolean indicating whether img1 is on the left (True) or right (False).

        Returns:
        - result_image: The stitched image.
        """
        # Normalize input images
        img1 = normalize_img(img1)
        img2 = normalize_img(img2)

        # Get dimensions of img1
        h1, w1 = img1.shape[:2]

        # Create an identity transformation matrix with horizontal offset
        m = np.eye(3)
        m[0, 2] = x_offset

        # Warp img1 using the transformation matrix
        img1_warped = cv2.warpPerspective(img1, m, (w1 + x_offset, h1))

        # Initialize the stitched image with zeros
        stitched_img = np.zeros(img2.shape)

        # Place the warped img1 on the left or right side of the stitched image
        if img1_left:
            stitched_img[:, :w1 + x_offset] = img1_warped
        else:
            stitched_img[:, :w1] = img1_warped

        # Create masks for pixels with values in both images
        mask_both = np.all((stitched_img != 0, img2 != 0), axis=0)

        # Create a new image by taking the mean of pixels with values in both images
        mean_image_both = np.zeros_like(img2, dtype=np.float32)
        mean_image_both[mask_both] = (stitched_img[mask_both] + img2[mask_both]) / 2

        # Create a new image by adding other pixels together
        result_image = stitched_img + img2

        # Combine the two images by replacing pixels with mean values
        result_image[mask_both] = mean_image_both[mask_both]

        return result_image

    def find_bounding_box(self, img):
        """
        Find the bounding box of the main object in a binary or grayscale image.

        Parameters:
        - img: Input image (binary or grayscale).

        Returns:
        - x: X-coordinate of the top-left corner of the bounding box.
        - y: Y-coordinate of the top-left corner of the bounding box.
        - w: Width of the bounding box.
        - h: Height of the bounding box.
        """

        # Convert the image to grayscale if it's in RGB format
        if len(img.shape) > 2:
            gray = rgb2gray(img)
        else:
            gray = img.copy()

        # Convert the grayscale image to a binary mask
        binary_mask = cv2.threshold(gray * 255, 1, 255, cv2.THRESH_BINARY)[1]
        binary_mask = binary_mask.astype(np.uint8)

        # Make a copy of the binary mask
        binary_mask = binary_mask.copy()

        # Define a kernel for morphological closing to fill small holes
        closing_kernel = np.ones((15, 15), np.uint8)

        # Apply closing to the binary mask
        result_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (main object)
        max_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(max_contour)

        # Get the initial mask at bounding box coordinates
        mask = result_mask[y:y+h, x:x+w]

        # Iterate to refine the bounding box by adjusting its position
        while np.any(mask == 0):
            # Save the previous state of the mask
            prev_mask = mask.copy()

            # Adjust the bounding box
            x += 2
            y += 2
            w -= 4  # Decrease width by 4 to adjust for both sides
            h -= 4  # Decrease height by 4 to adjust for both top and bottom
            mask = result_mask[y:y+h, x:x+w]

            # Check if the adjustment changed the mask
            if np.array_equal(mask, prev_mask):
                break

        return x, y, w, h
    
    def stitch_backward(self, current_image, previous_image, accumulated_transform_matrix, offset=0, is_left=False):
        """
        Stitches the current image onto the previous image using the accumulated affine transformation matrix.

        Args:
        - current_image (numpy.ndarray): The image to be stitched onto the previous image.
        - previous_image (numpy.ndarray): The base image onto which the current image will be stitched.
        - accumulated_transform_matrix (numpy.ndarray): The accumulated affine transformation matrix.
        - offset (int, optional): Offset for stitching (default is 0).
        - is_left (bool, optional): Flag indicating whether the current image is on the left (default is False).

        Returns:
        - numpy.ndarray: The stitched image.
        """
        # Calculate the size of the output image
        output_width = previous_image.shape[1] + current_image.shape[1]
        output_height = previous_image.shape[0]
        output_size = (output_width, output_height)

        # Apply the accumulated affine transformation to the current image
        transformed_current_image = cv2.warpPerspective(current_image, accumulated_transform_matrix, output_size, borderMode=cv2.BORDER_TRANSPARENT)

        # Stitch the previous and transformed current images
        stitched_image = self.stitch_two_images(previous_image, transformed_current_image, offset, is_left)

        return stitched_image

    def stitch_multiple_images(self, img_splits, affine_matrices):
        """
        Stitches multiple images starting from the middle image in a list using homography matrix.

        Args:
        - img_splits (list): List of images to be stitched.
        - affine_matrices (list): List of affine transformation matrices corresponding to the images.

        Returns:
        - numpy.ndarray: The final stitched image.
        """
        # Calculate the index of the middle image
        k = len(img_splits) // 2 - 1
        num_of_imgs = len(img_splits)
        
        # Initialize reference image and accumulated matrices
        ref_img = img_splits[k]
        a_m_prev = np.eye(3)  # Identity matrix for the previous affine transformation
        a_m_next = np.eye(3)  # Identity matrix for the next affine transformation
        
        for i in range(k + 1):
            if k - i - 1 >= 0:
                prev_img = img_splits[k - i - 1]
                
                # Accumulate the previous affine transformation matrix
                a_m_prev = np.dot(a_m_prev, affine_matrices[k - i - 1])
                
                # Adjust for negative offset in x-direction
                if a_m_prev[0, 2] < 0:
                    offset_x = int(abs(a_m_prev[0, 2]))
                    a_m_prev[0, 2] += offset_x
                else:
                    offset_x = 0
                
                # Stitch the previous image onto the reference image
                ref_img = self.stitch_backward(prev_img, ref_img, a_m_prev, offset_x, True)
            else:
                offset_x = 0
            
            if k + i + 1 < num_of_imgs:
                next_img = img_splits[k + i + 1]
                
                # Accumulate the inverse of the next affine transformation matrix
                a_m_next = np.dot(a_m_next, np.linalg.inv(affine_matrices[k + i]))
                
                # Adjust for offset in x-direction
                a_m_next[0, 2] += offset_x
                
                # Stitch the next image onto the reference image
                ref_img = self.stitch_backward(next_img, ref_img, a_m_next, 0)
        
        return ref_img

    def get_sift_descriptors(self, image_splits, num_downsamples = 0):
        """
        Get SIFT descriptors for multiple images and store them in a list.

        Args:
        - image_splits (List[np.ndarray]): List of grayscale images to extract SIFT descriptors from.
        - num_downsamples (int, optional): Number of downsampling levels for SIFT (default is 0).

        Returns:
        - List: List of SIFT descriptors for each image.
        """
        sift_data = []
        sift_extractor = SIFT(num_downsamples=num_downsamples)

        for image in image_splits:
            sift_summary = sift_extractor.extract_sift_data(image)
            sift_data.append(sift_summary)

        return sift_data
    
    def compute_affine_matrices(self, gray_images, sift_data):
        """
        Compute affine matrices between consecutive images in a list.

        Args:
        - gray_img_splits: List of grayscale images.
        - sift_data: List of SIFT descriptors for each image.

        Returns:
        - np.ndarray: Array of affine matrices.
        """
        affine_matrices = []
        model = NearestNeighbors(n_neighbors=1)

        for i in range(1, len(gray_images)):
            img1 = np.uint8(gray_images[i-1] * 255)
            img2 = np.uint8(gray_images[i] * 255)

            src_sift = sift_data[i-1]
            dst_sift = sift_data[i]

            model.fit(dst_sift["Descriptor"].tolist())

            a_m = self.compute_homography_matrix(src_sift, dst_sift, model, 1)  # Assuming you have a function to compute homography matrix
            affine_matrices.append(a_m)

        return np.array(affine_matrices)
    
    def get_panorama(self, images_list_rgb):
        '''returns panorama image stitched from ordered image sequence in images_list_rgb.'''
        # convert rgb images to gray
        gray_images = [rgb2gray(img) for img in images_list_rgb] 
        
        # get sift descriptor for all images
        sift_data = self.get_sift_descriptors(gray_images)
        
        # get affine matrices between each two consecutive images
        affine_matrices = self.compute_affine_matrices(gray_images, sift_data)
        
        # get stitched image
        stitched_image = self.stitch_multiple_images(images_list_rgb, affine_matrices)
        
        # crop image and remove black border
        x, y, w, h = self.find_bounding_box(stitched_image)
        cropped_img = stitched_image[y:y+h, x:x+w]
        
        return cropped_img