import numpy as np
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load the trained KNN model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def process_image(image_path, model_path, patch_size=32, debug=True):
    """
    Process an image by:
    1. Dividing it into patches
    2. Classifying each patch
    3. Applying appropriate filter (sharpening or blurring)
    4. Reconstructing the image with blending
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for visualization of classifications
    classification_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    # Create a counter map to track how many times each pixel is classified
    classification_count = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale for both feature extraction AND processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create color version for output visualization
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Create output image
    output_img = np.zeros_like(img_color, dtype=np.float32)
    
    # Create importance weight mask (will be filled during processing)
    weight_mask = np.zeros((height, width), dtype=np.float32)
    
    # Track which patches were classified as important for debugging
    important_count = 0
    total_patches = 0
    
    # For handling edge cases, ensure we cover the entire image
    x_positions = list(range(0, width - patch_size + 1, patch_size//2))  # 50% overlap
    if width > patch_size and (width - patch_size) % (patch_size//2) != 0:
        x_positions.append(width - patch_size)
        
    y_positions = list(range(0, height - patch_size + 1, patch_size//2))  # 50% overlap
    if height > patch_size and (height - patch_size) % (patch_size//2) != 0:
        y_positions.append(height - patch_size)
    
    # Save some patches for debugging
    debug_patches = {
        'important': [],
        'non_important': []
    }
    
    # Create a blending mask with Gaussian falloff
    blend_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
    
    # Generate a smooth Gaussian-like falloff from center (1.0) to edges (0.05)
    center_y, center_x = patch_size // 2, patch_size // 2
    for y in range(patch_size):
        for x in range(patch_size):
            # Calculate normalized distance from center (0-1 range)
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / (patch_size / 2)
            # Apply Gaussian-like falloff but keep some minimum weight (0.05)
            blend_mask[y, x] = 0.05 + 0.95 * np.exp(-1.5 * dist * dist)
    
    # Process each patch
    for y in y_positions:
        for x in x_positions:
            # Make sure we don't go out of bounds
            if y + patch_size > height or x + patch_size > width:
                continue
                
            # Extract patch (grayscale)
            patch = img_gray[y:y+patch_size, x:x+patch_size].copy()
            
            # Check if patch is valid
            if patch.size == 0:
                continue
                
            # Resize patch to exact 32x32 to match training
            patch_resized = cv2.resize(patch, (32, 32))
            
            # Save some patches for debugging (first 5 of each type)
            if len(debug_patches['important']) < 5 or len(debug_patches['non_important']) < 5:
                os.makedirs("debug_patches", exist_ok=True)
            
            # Prepare patch for classification (flatten)
            features = patch_resized.flatten().reshape(1, -1)
            
            # Scale features
            features_scaled = (features - np.mean(features)) / (np.std(features) + 1e-10)
            
            # Classify the patch
            prediction = model.predict(features_scaled)[0]
            
            # Save debug patches
            if len(debug_patches['important' if prediction == 1 else 'non_important']) < 5:
                patch_type = 'important' if prediction == 1 else 'non_important'
                debug_patches[patch_type].append(patch_resized)
                cv2.imwrite(f"debug_patches/{patch_type}_{len(debug_patches[patch_type])}.png", patch_resized)
            
            total_patches += 1
            
            # Extract the patch from the original color image
            color_patch = img_color[y:y+patch_size, x:x+patch_size].copy()
            
            # Process based on classification
            if prediction == 1:  # Important
                important_count += 1
                
                # Exactly as in the original code:
                # 1. FIRST increase intensity by 50% (before sharpening)
                color_patch = cv2.convertScaleAbs(color_patch * 1.4)  # Brighter by 50%
                
                # 2. Apply Laplacian sharpening with reduced effect (0.2 instead of 0.3)
                for c in range(3):  # Process each channel
                    channel = color_patch[:,:,c]
                    laplacian = cv2.Laplacian(channel, cv2.CV_64F)
                    sharpened = channel + 0.1 * cv2.convertScaleAbs(laplacian)  # Reduced sharpening
                    color_patch[:,:,c] = np.clip(sharpened, 0, 255)
                
                # Update classification map with 1.0 for important patches
                classification_map[y:y+patch_size, x:x+patch_size] = 1.0
                classification_count[y:y+patch_size, x:x+patch_size] += 1.0

            else:  # Non-important
                # Exactly as in the original code:
                # 1. Apply Gaussian blur
                color_patch = cv2.GaussianBlur(color_patch, (11, 11), 7)
                
                # 2. Reduce brightness by 50%
                color_patch = cv2.convertScaleAbs(color_patch)
                
                # Update classification count but keep classification_map as is (zeros)
                classification_map[y:y+patch_size, x:x+patch_size] = 0.0
                classification_count[y:y+patch_size, x:x+patch_size] += 1.0
            
            # Apply blending mask to the patch
            color_patch_float = color_patch.astype(np.float32)
            blended_patch = np.zeros_like(color_patch_float)
            for c in range(3):
                blended_patch[:,:,c] = color_patch_float[:,:,c] * blend_mask
            
            # Add to output image
            output_img[y:y+patch_size, x:x+patch_size] += blended_patch
            
            # Add blend mask to weight accumulator
            weight_mask[y:y+patch_size, x:x+patch_size] += blend_mask
    
    # Print classification distribution for debugging
    if debug:
        print(f"Patches classified as important: {important_count}/{total_patches} ({important_count/total_patches*100:.1f}%)")
    
    # Normalize by weight mask (avoid division by zero)
    weight_mask[weight_mask == 0] = 1.0
    
    # Apply normalization
    for c in range(3):
        output_img[:,:,c] = output_img[:,:,c] / weight_mask
    
    # Convert back to uint8
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    
    # Normalize the classification map - now each pixel has a value between 0 and 1
    # representing the fraction of patches that classified it as important
    classification_count = np.maximum(classification_count, 1.0)  # Avoid division by zero
    normalized_map = classification_map / classification_count
    
    # Convert to uint8 with the desired ratio of important vs non-important
    # Set threshold to match approximately the same ratio as the patch classifier
    target_ratio = important_count / total_patches
    
    # Simpler threshold calculation:
    # Sort the unique values and find the one that gives the right balance
    unique_values = np.unique(normalized_map)
    if len(unique_values) > 1:
        # Start with a threshold of 0.5
        threshold = 0.5
        
        # Convert to binary with the current threshold
        binary_map = (classification_map >= threshold).astype(np.uint8) * 255
        
        # Calculate the current ratio of white pixels
        white_ratio = np.sum(binary_map > 0) / binary_map.size
        
        # If the white ratio is too far from target, adjust threshold
        if abs(white_ratio - target_ratio) > 0.1:
            # Try a simpler approach - use 0.3 as threshold which typically gives good results
            threshold = 0.3
            binary_map = (normalized_map >= threshold).astype(np.uint8) * 255
    else:
        # If all values are the same, use the target ratio to determine
        binary_map = np.zeros_like(normalized_map, dtype=np.uint8)
        if unique_values[0] > 0:
            binary_map = np.ones_like(normalized_map, dtype=np.uint8) * 255
    
    return output_img, binary_map, debug_patches, blend_mask

def visualize_results(original_img, processed_img, classification_map, debug_patches, blend_mask, output_path="output"):
    """
    Visualize and save the original image, processed image, and classification map
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save processed image
    cv2.imwrite(os.path.join(output_path, "processed_image.jpg"), processed_img)
    
    # Save classification map
    cv2.imwrite(os.path.join(output_path, "classification_map.jpg"), classification_map)
    
    # Calculate histogram of classification map for debugging
    hist = cv2.calcHist([classification_map], [0], None, [256], [0, 256])
    zero_pixels = hist[0][0]
    white_pixels = hist[255][0] if len(hist) > 255 else 0
    total_pixels = classification_map.shape[0] * classification_map.shape[1]
    
    print(f"Classification Map Stats:")
    print(f"  Black pixels (non-important): {zero_pixels} ({zero_pixels/total_pixels*100:.1f}%)")
    print(f"  White pixels (important): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    plt.title("Processed Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(classification_map, cmap='gray')
    plt.title(f"Classification Map\nWhite = Important")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "comparison.jpg"))
    
    # Visualize debug patches
    if debug_patches['important'] or debug_patches['non_important']:
        plt.figure(figsize=(15, 10))
        
        # Plot important patches
        for i, patch in enumerate(debug_patches['important'][:5]):
            if i < len(debug_patches['important']):
                plt.subplot(2, 5, i+1)
                plt.imshow(patch, cmap='gray')
                plt.title(f"Important {i+1}")
                plt.axis("off")
        
        # Plot non-important patches
        for i, patch in enumerate(debug_patches['non_important'][:5]):
            if i < len(debug_patches['non_important']):
                plt.subplot(2, 5, 5+i+1)
                plt.imshow(patch, cmap='gray')
                plt.title(f"Non-important {i+1}")
                plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "debug_patches.jpg"))
    
    # Visualize blending mask
    plt.figure(figsize=(6, 6))
    img_plot = plt.imshow(blend_mask, cmap='viridis')
    plt.colorbar(img_plot, label='Weight')
    plt.title('Patch Blending Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "blend_mask.jpg"))
    
    plt.close('all')  # Close all figures to avoid memory issues

def main():
    # Input parameters
    image_path="image_10003_label_Moderate Demented.jpg"
    model_path = "0.90-acc-knn_patch_classifier.pkl"
    
    # Process the image
    print(f"Processing image: {image_path}")
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Failed to load image: {image_path}")
            return
            
        print(f"Image size: {original_img.shape}")
        processed_img, classification_map, debug_patches, blend_mask = process_image(image_path, model_path)
        
        # Visualize and save results
        visualize_results(original_img, processed_img, classification_map, debug_patches, blend_mask)
        print("Processing complete! Results saved to the 'output' directory.")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()