import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from image")

    # Basic command line arguments
    parser.add_argument('image', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes (default: 1)')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()