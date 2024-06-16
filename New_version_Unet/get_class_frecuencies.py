import os
import cv2
import numpy as np

def count_pixels(image, colors):
    counts = {tuple(color): 0 for color in colors}
    total_pixels = image.shape[0] * image.shape[1]
    for color in colors:
        counts[tuple(color)] = np.sum(np.all(image == color, axis=-1))
    return counts, total_pixels

def process_images(image_dir, colors):
    color_counts = {tuple(color): 0 for color in colors}
    total_pixels = 0

    image = cv2.imread(image_dir)
    if np.all(image == [85, 85, 85]):
        return color_counts, total_pixels
    
    color_counts, total_pixels = count_pixels(image, colors)

    return color_counts, total_pixels

def main():
    image_dir = 'path_to_masks'  # Path to directory containing all images
    colors = [
        np.array([0, 0, 0]),            # BLACK
        np.array([85, 85, 85]),         # (85,85,85)
        np.array([170, 170, 170]),      # (170, 170, 170)
        np.array([255, 255, 255])       # (255,255,255)
    ]

    groups = {}
    total_folder_pixels = 0  # To accumulate total pixels across all images
    global_color_counts = {tuple(color): 0 for color in colors}

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            prefix = filename[:20] 
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(filename)

    for group_prefix, filenames in groups.items():
        group_weight = len(filenames) / 154 # change number for the total amount of patches created
        print(f"Group: {group_prefix}")
        print(f"Number of files: {len(filenames)}")
        
        group_color_counts = {tuple(color): 0 for color in colors}
        total_group_pixels = 0
        
        for filename in filenames:
            file_path = os.path.join(image_dir, filename)
            color_counts, total_pixels = process_images(file_path, colors)
            total_group_pixels += total_pixels
            
            # Accumulate color counts for the group
            for color, count in color_counts.items():
                group_color_counts[color] += count
        
        print("Color Percentage for Group:")
        for color, count in group_color_counts.items():
            percentage = (count / total_group_pixels) * 100
            global_color_counts[color] += group_weight * percentage
            print(f"Color {color}: {percentage:.2f}%")
        print()
        
        total_folder_pixels += total_group_pixels  # Accumulating total pixels

    print("Global Color Percentage for Folder:")
    for color, percentage in global_color_counts.items():
        print(f"Color {color}: {percentage:.2f}%")
    print()

if __name__ == "__main__":
    main()
