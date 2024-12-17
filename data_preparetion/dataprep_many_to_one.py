import os
import cv2
import numpy as np

# Directories for the T1, T2, and PD images
# Replace these with the paths to your image directories
#For T1_T2__PD make first T1, second T2, third PD
#For T1_PD__T2 make first T1, second PD, third T2
#For T2_PD__T1 make first T2, second PD, third T1
#For T1_T2__Flair make first T1, second T2, third Flair
#For T1_Flair__T2 make first T1, second Flair, third T2
#For T2_Flair__T1 make first T2, second Flair, third T1

input_dir_first = 'yourfirstpath'
input_dir_second = 'yoursecondpath'
input_dir_third = 'yourthirdpath'
output_dir = 'youroutputpath'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get the list of files from one folder (assuming same filenames in all folders)
file_list = os.listdir(input_dir_first)

# Process each file
for filename in file_list:
    t1_path = os.path.join(input_dir_first, filename)
    t2_path = os.path.join(input_dir_second, filename)
    pd_path = os.path.join(input_dir_third, filename)

    # Check if all required files exist
    if not (os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(pd_path)):
        print(f"Skipping {filename} as corresponding images are missing.")
        continue

    # Read the images
    t1_image = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
    t2_image = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
    pd_image = cv2.imread(pd_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 256x256 if necessary
    t1_image_resized = cv2.resize(t1_image, (256, 256))
    t2_image_resized = cv2.resize(t2_image, (256, 256))
    pd_image_resized = cv2.resize(pd_image, (256, 256))

    # Create the second channel (256x512 with T2 and PD side-by-side)
    second_channel = np.hstack((t2_image_resized, pd_image_resized))

    # Pad T1 image to 256x512 (pad on the right side with zeros)
    t1_image_padded = cv2.copyMakeBorder(t1_image_resized, 0, 0, 0, 256, cv2.BORDER_CONSTANT, value=0)

    # Create the third channel (empty array)
    third_channel = np.zeros((256, 512), dtype=np.uint8)

    # Stack the channels correctly
    combined_image = np.dstack((third_channel, second_channel, t1_image_padded))

    # Save the combined image as a new file
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, combined_image)

print(f"Processed images saved to {output_dir}")
