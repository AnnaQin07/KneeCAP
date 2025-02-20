from post_utils import bone_boxing, crop_bone, bilinear_interpolation, find_inclose_circle, shape_check
from landmarks import locate_femur_head_v0, locate_condylar_features, bilinear_interpolation, find_inclose_circle, shape_check,locate_tibia_implant_plateau_features

import cv2
import numpy as np
import matplotlib.pyplot as plt

# upload mask
image_path = r"D:\document\anu_project\project\dataset\dataset\For_training\finetune\split\train\hd_masks\Ac-IM-0024-0005_l.png"
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Get unique pixel values in the mask [  0  76 149(femur)]
processed_mask = np.zeros_like(mask)
processed_mask[mask == 149] = 1


def test_bone_boxing(image_path):
    """
    Test the bone_boxing function by loading the mask,
    finding the bounding box of the bone, and displaying the result.

    Args:
        image_path (str): The path to the image mask.

    Returns:
        None
    """
    # Load the mask
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Process the mask to isolate the femur (pixel value 149)
    processed_mask = np.zeros_like(mask)
    processed_mask[mask == 149] = 1

    # Get the bounding box using the bone_boxing function
    x1, y1, x2, y2 = bone_boxing(processed_mask)

    # Display the original mask and draw the bounding box
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')

    # Draw the bounding box
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title(f"Bounding Box: ({x1}, {y1}), ({x2}, {y2})")
    plt.axis('off')
    plt.show()

# test_bone_boxing(image_path)

def test_locate_condylar_features(mask, processed_mask):
    """
    Tests the correctness of the locate_condylar_features function by plotting the original and processed mask
    and displaying the condylar features on the original mask.

    Args:
        mask (ndarray): Original mask.
        processed_mask (ndarray): Processed mask used to locate condylar features.
    """
    # Plot the original and processed mask side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title('Original Mask')
    axs[0].axis('off')

    axs[1].imshow(processed_mask, cmap='gray')
    axs[1].set_title('Processed Mask (2nd Largest Region)')
    axs[1].axis('off')

    plt.show()

    # Call the function to locate condylar features
    features = locate_condylar_features(processed_mask)

    # Plot the original mask and mark the condylar features
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='gray')

    if features['condylar_midway'] is not None:
        plt.scatter(features['condylar_midway'][0], features['condylar_midway'][1], color='r', marker='o',
                    label='Condylar Midway')
    if features['condylar_left'] is not None:
        plt.scatter(features['condylar_left'][0], features['condylar_left'][1], color='g', marker='x',
                    label='Condylar Left')
    if features['condylar_right'] is not None:
        plt.scatter(features['condylar_right'][0], features['condylar_right'][1], color='b', marker='x',
                    label='Condylar Right')

    plt.legend()
    plt.title("Condylar Features")
    plt.axis('off')
    plt.show()
# test_locate_condylar_features(mask, processed_mask)


def test_tibia_implant():
    l=[]
    tibia_mask = np.zeros_like(mask)
    tibia_mask[mask == 76] = 1
    bbx=bone_boxing(tibia_mask)
    roi=crop_bone(tibia_mask,bbx)
    interg_x=np.sum(tibia_mask, axis=1)
    features = locate_tibia_implant_plateau_features(roi,interg_x,bbx)

    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='gray')

    if features['center'] is not None:
        plt.scatter(features['center'][0], features['center'][1], color='r', marker='o',
                    label='center')
    if features['left_terminal'] is not None:
        plt.scatter(features['left_terminal'][0], features['left_terminal'][1], color='g', marker='x',
                    label='left_terminal')
    if features['right_terminal'] is not None:
        plt.scatter(features['right_terminal'][0], features['right_terminal'][1], color='b', marker='x',
                    label='right_terminal')

    plt.legend()
    plt.title("Condylor Features")
    plt.axis('off')
    plt.show()