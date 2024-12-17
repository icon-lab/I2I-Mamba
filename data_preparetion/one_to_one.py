import os
import cv2
import numpy as np

# Paths to the two folders
# For T1__T2 make first T1, second T2
# For PD__T2 make first PD, second T2 and so on

folder1 = 'yourfirstpath'  # Replace with the path to your first folder
folder2 = 'yoursecondpath'  # Replace with the path to your second folder

# Output folder to save combined images
output_folder = 'youroutputpath'  # Replace with the path to your output folder
os.makedirs(output_folder, exist_ok=True)

# Get the list of filenames in both folders (assuming they have the same filenames)
file_list1 = sorted(os.listdir(folder1))
file_list2 = sorted(os.listdir(folder2))

# Process each file
for filename1, filename2 in zip(file_list1, file_list2):
    path1 = os.path.join(folder1, filename1)
    path2 = os.path.join(folder2, filename2)
    
    # Read the images in grayscale mode
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    
    # Check if images are read correctly
    if image1 is None or image2 is None:
        print(f"Skipping {filename1} or {filename2} as one of the images is missing.")
        continue

    # Resize both images to 256x256
    image1_resized = cv2.resize(image1, (256, 256))
    image2_resized = cv2.resize(image2, (256, 256))

    # Combine the two images side by side (256x512)
    combined_image = np.hstack((image1_resized, image2_resized))

    # Save the combined image
    output_path = os.path.join(output_folder, f"combined_{filename1}")
    cv2.imwrite(output_path, combined_image)

print(f"Combined images saved to {output_folder}")
