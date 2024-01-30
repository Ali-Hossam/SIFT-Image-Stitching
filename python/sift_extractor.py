import cv2
import numpy as np
import pandas as pd
from numba import jit
from skimage.transform import rotate
from general_functions import get_hist, sobel_numba, compute_weighted_histogram

class SIFT:
    def __init__(self, sigma=1.2, k=np.sqrt(1.8), num_downsamples=3, doG_thresh=0.003, r=10):
        self.doG_thresh = doG_thresh
        self.r = r
        self.sigmas = [sigma, k * sigma, k**2 * sigma, k**3 * sigma, k**4 * sigma, k**5 * sigma]
        self.downsampling_lvls = [1.0]
        
        for i in range(num_downsamples):
            self.downsampling_lvls.append(np.sqrt(2) * self.downsampling_lvls[-1])


    def downsample_img(self, img):
        """
        Generates a list of images by downsampling the input image at different levels.

        Parameters:
        - img: Input image (NumPy array).
        - lvls: List of downsampling levels.

        Returns:
        - generated_imgs: List of images downsampled at different levels.
        """
        lvls = self.downsampling_lvls
        generated_imgs = [cv2.resize(img, (0, 0), fx=1.0/lvl, fy=1.0/lvl) for lvl in lvls]
        return generated_imgs

    def create_octave(self, img):
        """
        Generates a list of images by applying Gaussian blur to the input image at different sigma values.

        Parameters:
        - img: Input image (NumPy array).
        - sigmas: List of sigma values for Gaussian blurring.

        Returns:
        - generated_imgs: List of images blurred at different sigma values.
        """
        sigmas = self.sigmas
        blurred_images = [cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma) for sigma in sigmas]
        return blurred_images

    def get_doG(self, imgs_list):
        """
        Computes the Difference of Gaussians (doG) for a list of images.

        Parameters:
        - imgs_list: List of images.

        Returns:
        - doG_list: List of images representing the differences between consecutive scales.
        """
        doG_list = [imgs_list[i] - imgs_list[i - 1] for i in range(1, len(imgs_list))]
        return doG_list

    @staticmethod
    @jit(nopython=True)
    def get_keypoints(doG_list, thresh=0.003):  # check that image values are between 0 and 1
        """
        Detects potential keypoints in Difference of Gaussian (doG) images.

        Args:
        - doG_list (list of arrays): A list containing Difference of Gaussian images.
        - thresh (float): Threshold value for identifying potential keypoints (default: 0.55).

        Returns:
        - total_keypoints (list of lists of tuples): A list containing coordinates of identified keypoints for each doG image.

        Notes:
        - The function iterates through the doG images in 'doG_list' and identifies potential keypoints.
        - 'thresh' determines the sensitivity of keypoint detection.
        - Output 'total_keypoints' contains lists of keypoints corresponding to each doG image.
        """
        candidate_keypoints = []
        H, W = doG_list[0].shape

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                wind1 = doG_list[0][y - 1: y + 2, x - 1: x + 2]
                wind2 = doG_list[1][y - 1: y + 2, x - 1: x + 2]
                wind3 = doG_list[2][y - 1: y + 2, x - 1: x + 2]

                if np.abs(doG_list[1][y, x]) > thresh:
                    if (doG_list[1][y, x] > max([np.max(wind1), np.max(wind3)]) or
                        doG_list[1][y, x] < min([np.min(wind1), np.min(wind3)])):

                        if (doG_list[1][y, x] == np.max(wind2) or doG_list[1][y, x] == np.min(wind2)):
                            candidate_keypoints.append((x, y))  # Swap x and y here
        
        return candidate_keypoints

    def filter_keypoints(self, img, keypoints, dxx, dyy, dxy, r=10, patch_size=5):
        """
        Filter keypoints based on the Hessian matrix determinant ratio.

        Parameters:
        - img: Input image.
        - keypoints: List of keypoints, where each keypoint is represented as a tuple (x, y).
        - dxx, dyy, dxy: Second derivatives of the image in x, y directions.
        - r: Parameter controlling the acceptance of keypoints based on Hessian determinant ratio.
            Keypoints with a ratio below ((r + 1)^2) / r are accepted.
        - patch_size: Size of the neighborhood patch for computing derivatives.

        Returns:
        List of filtered keypoints based on the Hessian determinant ratio.
        Each filtered keypoint is represented as a tuple (x, y).
        """
        filtered_keypoints = []

        for kp in keypoints:
            x, y = kp
            
            # Calculate the Hessian matrix at the keypoint location
            y_start, y_end = int(y - patch_size/2), int(y + patch_size/2)
            x_start, x_end = int(x - patch_size/2), int(x + patch_size/2)
            
            dxx_ = dxx[y_start:y_end, x_start:x_end]
            dyy_ = dyy[y_start:y_end, x_start:x_end]
            dxy_ = dxy[y_start:y_end, x_start:x_end]

            if dxx_.any():
                # Construct the Hessian matrix
                H = np.array([[np.sum(dxx_**2), np.sum(dxy_**2)],
                            [np.sum(dxy_**2), np.sum(dyy_**2)]])

                # Calculate the Hessian determinant ratio
                ratio = (H[0, 0] + H[1, 1])**2 / (np.linalg.det(H) + 1e-8)
                
                # Check if the keypoint satisfies the Hessian determinant ratio condition
                if(x < img.shape[1] - 1 and y < img.shape[0] - 1):
                    if ratio < ((r + 1)**2) / r:
                        filtered_keypoints.append((x, y))
        return filtered_keypoints

    def get_dominant_angles(self, filtered_keypoints, gradient_magnitude, gradient_orientation, patch_size=8):
        """
        Calculate dominant angles for filtered keypoints based on local gradient information.

        Parameters:
        - filtered_keypoints: List of filtered keypoints, where each keypoint is represented as a tuple (x, y).
        - gradient_magnitude: Gradient magnitude image.
        - gradient_orientation: Gradient orientation image.
        - patch_size: Size of the neighborhood patch for computing orientations.

        Returns:
        - orientation_data: Dictionary mapping each keypoint to a list of dominant angles.
        - new_filtered_keypoints: List of keypoints that passed the patch size check.
        """
        orientation_data = {}
        new_filtered_keypoints = []

        for kp in filtered_keypoints:
            x, y = kp
            y_start, y_end = int(y - patch_size / 2), int(y + patch_size / 2)
            x_start, x_end = int(x - patch_size / 2), int(x + patch_size / 2)
            patch_magnitude = gradient_magnitude[y_start:y_end, x_start:x_end]
            patch_orientation = gradient_orientation[y_start:y_end, x_start:x_end]

            if patch_magnitude.shape == (patch_size, patch_size):
                # Flatten the orientation and magnitude arrays
                flat_magnitude = patch_magnitude.flatten()
                flat_orientation = patch_orientation.flatten()

                # Calculate the weighted histogram of orientations
                hist, bins = np.histogram(flat_orientation, bins=36, range=(0, 360), weights=flat_magnitude)

                # Find the bin index with the maximum value
                max_bin_index = np.argmax(hist)

                # Convert the bin index to the corresponding orientation value
                max_orientation = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2

                orientation_data[kp[:2]] = [max_orientation]
                new_filtered_keypoints.append(kp)

                # Check for additional prominent angles (>0.8 times the peak value)
                hist = np.delete(hist, max_bin_index)
                other_bins = np.where(hist > 0.8 * max_orientation)[0]

                if len(other_bins) > 0:
                    orientation_data[kp[:2]].extend([(bins[bin] + bins[bin + 1]) / 2 for bin in other_bins])

        return orientation_data, new_filtered_keypoints


    def get_descriptor(self, doG_i, kp, rot_angle, patch_size=16, subarray_size=4):
        """
        Generates a sift descriptor for a keypoint in an image.

        Parameters:
        - doG_i (numpy.ndarray): The Difference of Gaussians (DoG) image from which the descriptor is computed.
        - kp (tuple): Keypoint coordinates (x, y) for which the descriptor is generated.
        - rot_angle (float): Rotation angle in degrees representing the dominant orientation of the keypoint.
        - patch_size (int, optional): Size of the square patch around the keypoint. Default is 16.
        - subarray_size (int, optional): Size of the subarrays within the patch. Default is 4.
        - func (function, optional): Function used to compute the feature vector for each subarray. Default is get_hist.

        Returns:
        - normalized_feature_vector (numpy.ndarray): Normalized feature vector representing the sift descriptor.

        Note:
        - The function checks boundaries to ensure that the generated patch around the keypoint is within the image limits.
        - The patch is rotated using the specified dominant angle before computing the feature vector.
        - The feature vector is obtained by applying the specified function to subarrays within the rotated patch.
        - The resulting feature vector is normalized to ensure invariance to scale and illumination changes.

        Example usage:
        descriptor = get_descriptor(doG_image, (x, y), rotation_angle)
        """

        half_patch = patch_size // 2
        x, y = kp

        # Check boundaries
        x_start = max(x - half_patch, 0)
        y_start = max(y - half_patch, 0)
        x_end = min(x + half_patch, doG_i.shape[1] - 1)
        y_end = min(y + half_patch, doG_i.shape[0] - 1)
        
        patch = doG_i[y_start:y_end, x_start:x_end]
        if patch.shape != (16, 16):
            return []

        # Rotate patch with the dominant angle
        patch = rotate(patch, angle=rot_angle, resize=False, preserve_range=True)

        feature_vector = []
        for row in range(0, patch_size, subarray_size):
            for col in range(0, patch_size, subarray_size):
                subarray = patch[row:min(row + subarray_size, patch_size), col:min(col + subarray_size, patch_size)]
                result = get_hist(subarray)
                feature_vector.extend(result)

        norm = np.linalg.norm(feature_vector)
        if norm:
            normalized_feature_vector = feature_vector / norm
        else:
            normalized_feature_vector = feature_vector

        return normalized_feature_vector


    def extract_sift_data(self, img):
        """
        Extracts sift keypoints and descriptors from an input image using the Scale-Invariant Feature Transform (sift) algorithm.

        Parameters:
        - img (numpy.ndarray): Input image in the form of a NumPy array.
        - sigmas (list): List of standard deviations for the Gaussian kernels used in creating the scale space.
        - downsampling_lvls (list, optional): List of downsampling levels for creating different sizes of the image. Default is [1].
        - doG_thresh (float, optional): Threshold for the Difference of Gaussians (DoG) values to identify keypoints. Default is 0.003.
        - r (int, optional): Radius for filtering keypoints based on the Hessian matrix. Default is 10.

        Returns:
        - sift_summary (pandas.DataFrame): DataFrame containing sift keypoints and descriptors with columns:
            - "Position": Keypoint position (x, y).
            - "Size": Keypoint size.
            - "Orientation": Dominant orientation of the keypoint.
            - "Descriptor": sift descriptor for the keypoint.
        """
        
        sift_summary = pd.DataFrame(columns=["Position", "Size", "Orientation", "Descriptor"])

        # Generate different sizes of the image
        downsampled_imgs = self.downsample_img(img)

        # Create scale space
        scale_space = []
        for i in range(len(downsampled_imgs)):
            octave = self.create_octave(downsampled_imgs[i])
            scale_space.append(octave)

        for s, imgs_list in enumerate(scale_space):
            doG_list = np.array(self.get_doG(imgs_list))

            for i in range(1, len(doG_list)-1):  # i is the scale (blur lvl)
                doG_i = doG_list[i].astype(np.float32)

                # Get keypoints of this doG
                keypoints = self.get_keypoints(np.array(doG_list), self.doG_thresh)

                # Calculate first and second derivatives of the image in both x, y directions
                dx = cv2.Sobel(doG_i, cv2.CV_64F, 1, 0, ksize=3)
                dy = cv2.Sobel(doG_i, cv2.CV_64F, 0, 1, ksize=3)
                dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
                dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
                dxy = cv2.Sobel(dy, cv2.CV_64F, 1, 0, ksize=3)

                # Calculate magnitude and direction for this doG img
                gradient_magnitude, gradient_orientation = cv2.cartToPolar(dx, dy, angleInDegrees=True)

                # Filter keypoints to include only corner points
                filtered_keypoints = self.filter_keypoints(doG_i, keypoints, dxx, dyy, dxy, r=self.r)

                orientation_data, filtered_keypoints = self.get_dominant_angles(filtered_keypoints, gradient_magnitude, gradient_orientation)


                for kp in filtered_keypoints:
                    for angle in orientation_data[kp[:2]]:
                        descriptor = self.get_descriptor(doG_i, kp, angle)
                        if len(descriptor):
                            data = [kp, 1/self.downsampling_lvls[s], angle, descriptor]
                            sift_summary.loc[len(sift_summary)] = data
        return sift_summary
