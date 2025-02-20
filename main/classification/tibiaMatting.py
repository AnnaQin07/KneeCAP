import cv2
import numpy as np
import os

# Define folder paths
origin_path = 'E:/ANU/24s2/8715/DATA/dataset/For_training/pretrain/split/train/images'  # Path to the original images folder
seg_path = 'E:/ANU/24s2/8715/DATA/dataset/For_training/pretrain/split/train/hd_masks'  # Path to the segmentation results folder
output_path = 'E:/ANU/24s2/8715/DATA/dataset/matting'  # Path to the folder where the result will be saved
classifi_path = 'E:/ANU/24s2/8715/DATA/images_multiclassification/2_implants'  # Path to the folder for checking if the image exists

# Ensure folder_c exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Iterate over the images in folder_a and folder_b
for img_name in os.listdir(origin_path):
    img_path_a = os.path.join(origin_path, img_name)  # Path to the original image
    img_path_b = os.path.join(seg_path, img_name)  # Path to the segmentation result
    img_path_d = os.path.join(classifi_path, img_name)  # Path to the image in folder_d

    # If the image does not exist in folder_d, skip this image
    if not os.path.exists(img_path_d):
        print(f"{img_name} not found in folder_d, skipping...")
        continue

    # Read the original image and segmentation result
    original_img = cv2.imread(img_path_a)
    segmentation_img = cv2.imread(img_path_b)

    # Convert the segmentation image to HSV color space
    hsv_segmentation = cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the red area (can be adjusted based on the image)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Create a mask to select the red area
    mask = cv2.inRange(hsv_segmentation, lower_red, upper_red)

    # Create a four-channel image (original image + Alpha channel)
    bgr_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)

    # Apply the mask to the Alpha channel to set the non-red areas as transparent
    bgr_image[:, :, 3] = mask  # Use the mask as the Alpha channel

    # Save the extracted result with a transparent background as a PNG image
    save_path = os.path.join(output_path, img_name.split('.')[0] + '.png')
    cv2.imwrite(save_path, bgr_image)

    print(f"Transparent tibia from {img_name} saved to {save_path}")
