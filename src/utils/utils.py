from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d

def smooth_predictions(predicted_classes, probs, window_size= 5):
    """
    Smooth the predicted classes and probabilities using a moving average filter.
    
    Args:
        predicted_classes (List[int]): List of predicted class indices.
        probs (np.ndarray): Array of probabilities for each class.
        window_size (int): Size of the moving average window.
        
    Returns:
        Tuple[List[int], np.ndarray]: Smoothed predicted classes and probabilities.
    """
    # Smooth predicted classes
    smoothed_classes = uniform_filter1d(predicted_classes, size=window_size, mode='nearest').astype(int).tolist()
    
    # Smooth probabilities
    smoothed_probs = gaussian_filter1d(probs, sigma=window_size, axis=0)
    
    return smoothed_classes, smoothed_probs