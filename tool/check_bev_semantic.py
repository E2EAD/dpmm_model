import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_visualize_semantic_mask(image_path):
    # Read the image with IMREAD_UNCHANGED to preserve original dtype and channels
    mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Analyze the mask properties
    print(f"Image path: {image_path}")
    print(f"dtype: {mask.dtype}")  # Shows the data type (e.g., uint8, uint16)
    print(f"Shape: {mask.shape}")  # Shows dimensions (height, width, channels)
    
    if len(mask.shape) == 2:
        print("This is a single-channel (grayscale) mask")
        print(f"Unique values: {np.unique(mask)}")
        print(f"Value range: {np.min(mask)} to {np.max(mask)}")
    elif len(mask.shape) == 3:
        print(f"This is a {mask.shape[2]}-channel mask")
        # For each channel, show value range
        for i in range(mask.shape[2]):
            channel = mask[:, :, i]
            print(f"Channel {i} range: {np.min(channel)} to {np.max(channel)}")
    
    # Visualization function based on your example
    def encode_semantic_to_rgb(semantic_array):
        """Convert semantic mask to RGB for visualization"""
        # Handle different input formats
        if len(semantic_array.shape) == 2:  # Single-channel mask
            # Create a color map for class indices
            num_classes = int(np.max(semantic_array)) + 1
            print(f"Detected {num_classes} semantic classes")
            
            # Create a random color map for each class
            np.random.seed(42)
            color_map = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)
            
            # Apply color map
            if num_classes > 0:
                colored_mask = color_map[semantic_array]
                return colored_mask
            return cv2.cvtColor((semantic_array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        elif len(semantic_array.shape) == 3:  # Multi-channel mask (c, h, w) or (h, w, c)
            # Check channel order - convert to (c, h, w) if needed
            if semantic_array.shape[0] < semantic_array.shape[1] and semantic_array.shape[0] < semantic_array.shape[2]:
                # Already in (c, h, w) format
                c, h, w = semantic_array.shape
            else:
                # Convert from (h, w, c) to (c, h, w)
                semantic_array = np.transpose(semantic_array, (2, 0, 1))
                c, h, w = semantic_array.shape
            
            print(f"Processing {c} semantic channels")
            
            # Create output RGB image
            rgb = np.zeros((3, h, w), dtype=np.uint8)
            
            # Process each channel group as in your example
            for i in range(c):
                if 0 <= i <= 4:  # First 5 channels -> R
                    bit_pos = 7 - (i % 5)
                    rgb[0] = rgb[0] | ((semantic_array[i] > 0).astype(np.uint8) << bit_pos)
                elif 5 <= i <= 9:  # Next 5 channels -> G
                    bit_pos = 7 - ((i-5) % 5)
                    rgb[1] = rgb[1] | ((semantic_array[i] > 0).astype(np.uint8) << bit_pos)
                elif 10 <= i <= 14:  # Next 5 channels -> B
                    bit_pos = 7 - ((i-10) % 5)
                    rgb[2] = rgb[2] | ((semantic_array[i] > 0).astype(np.uint8) << bit_pos)
            
            # Transpose back to (h, w, 3) for OpenCV
            return np.transpose(rgb, (1, 2, 0))
        
        return None
    
    # Create visualization
    if mask is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to proper format for encoding
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        # This might already be an encoded RGB mask - check values
        print("Image appears to be already encoded as RGB. Decoding for better visualization...")
        # Create a visualization that highlights active channels
        r, g, b = mask[:, :, 2], mask[:, :, 1], mask[:, :, 0]  # Convert BGR to RGB
        vis_r = (r > 0).astype(np.uint8) * 255
        vis_g = (g > 0).astype(np.uint8) * 255
        vis_b = (b > 0).astype(np.uint8) * 255
        colored_mask = cv2.merge([vis_b, vis_g, vis_r])
    else:
        # Standard case - process as semantic mask
        colored_mask = encode_semantic_to_rgb(mask)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if len(mask.shape) == 2:
        plt.imshow(mask, cmap='gray')
    else:
        # Convert from BGR to RGB for display
        display_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if mask.shape[2] == 3 else mask
        plt.imshow(display_mask)
    plt.title('Original Mask (may appear black)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
    plt.title('Visualized Semantic Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('semantic_visualization.png', dpi=300)
    print("Saved visualization as 'semantic_visualization.png'")
    
    # Show the plot
    plt.show()

# Example usage
analyze_and_visualize_semantic_mask('semantic_mask.png')
