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
    4. Reconstructing the image
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Create a scaler for preprocessing
    scaler = StandardScaler()
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy for visualization of classifications
    classification_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale for both feature extraction AND processing
    # This matches the preprocessing in your GradCAM script
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create color version for output visualization
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # Create output image
    output_img = np.zeros_like(img_color, dtype=np.float32)
    
    # Track which patches were classified as important for debugging
    important_count = 0
    total_patches = 0
    
    # For handling edge cases, ensure we cover the entire image
    x_positions = list(range(0, width - patch_size + 1, patch_size))
    if width > patch_size and (width - patch_size) % patch_size != 0:
        x_positions.append(width - patch_size)  # Add the last possible x position
        
    y_positions = list(range(0, height - patch_size + 1, patch_size))
    if height > patch_size and (height - patch_size) % patch_size != 0:
        y_positions.append(height - patch_size)  # Add the last possible y position
    
    # Track contribution for each pixel (for proper averaging in case of overlap)
    contribution_count = np.zeros((height, width, 3), dtype=np.float32)
    
    # Save some patches for debugging
    debug_patches = {
        'important': [],
        'non_important': []
    }
    
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
            
            # IMPORTANT: Scale with fixed values similar to what we'd expect during training
            # Gray images typically have values around 0-255, so standardize accordingly
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
                
                # 1. FIRST increase intensity by 50% (before sharpening)
                color_patch = cv2.convertScaleAbs(color_patch * 1.5)  # Brighter by 50%
                
                # 2. Apply Laplacian sharpening with reduced effect (0.2 instead of 0.3)
                for c in range(3):  # Process each channel
                    channel = color_patch[:,:,c]
                    laplacian = cv2.Laplacian(channel, cv2.CV_64F)
                    sharpened = channel + 0.2 * cv2.convertScaleAbs(laplacian)  # Reduced sharpening
                    color_patch[:,:,c] = np.clip(sharpened, 0, 255)
                
                classification_map[y:y+patch_size, x:x+patch_size] = 255

            else:  # Non-important
                # 1. Stronger Gaussian blur (kernel increased from (11,11) to (15,15))
                color_patch = cv2.GaussianBlur(color_patch, (5, 5), 7)
                
                # 2. Reduce brightness by 50%
                color_patch = cv2.convertScaleAbs(color_patch * 0.9)
            # Add processed patch to output image with tracking for blending
            output_img[y:y+patch_size, x:x+patch_size] += color_patch
            contribution_count[y:y+patch_size, x:x+patch_size] += 1
    
    # Print classification distribution for debugging
    if debug:
        print(f"Patches classified as important: {important_count}/{total_patches} ({important_count/total_patches*100:.1f}%)")
    
    # Normalize by contribution count (for overlapping regions)
    # Avoid division by zero
    contribution_count[contribution_count == 0] = 1
    
    # Divide each pixel by its contribution count
    for c in range(3):
        output_img[:,:,c] = output_img[:,:,c] / contribution_count[:,:,c]
    
    # Convert back to uint8
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    
    return output_img, classification_map, debug_patches

def visualize_results(original_img, processed_img, classification_map, debug_patches, output_path="output"):
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
    
    plt.close('all')  # Close all figures to avoid memory issues

def main():
    # Input parameters
    #image_path = "image_10003_label_Moderate Demented.jpg"  # Change to your image path
    image_path="image_10003_label_Moderate Demented_up.jpg"
    model_path = "0.90-acc-knn_patch_classifier.pkl"  # Change to your model path
    
    # Process the image
    print(f"Processing image: {image_path}")
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Failed to load image: {image_path}")
            return
            
        print(f"Image size: {original_img.shape}")
        processed_img, classification_map, debug_patches = process_image(image_path, model_path)
        
        # Visualize and save results
        visualize_results(original_img, processed_img, classification_map, debug_patches)
        print("Processing complete! Results saved to the 'output' directory.")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()