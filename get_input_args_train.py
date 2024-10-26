import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description='Train an image classifier model.')

    # Basic command line argument
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory of dataset (required)'
    )

    # Optional command line arguments

    # Checkpoints
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='.', 
        help='Directory to save checkpoints'
    )

    # Architecture
    parser.add_argument(
        '--arch', 
        type=str, 
        default='vgg16', 
        choices=['vgg16', 'densenet169'], 
        help='Choose architecture - vgg16 or densenet169 (default: vgg16)'
    )

    # Learning rate
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.001, 
        help='Set learning rate (default: 0.001)'
    )

    # Hidden units
    parser.add_argument(
        '--hidden_units', 
        type=int, 
        default=512, 
        help='Set hidden units for the classifier (default: 512)'
    )

    # Number of epochs
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=5, 
        help='Number of epochs (default: 5)'
    )

    # Use gpu
    parser.add_argument(
        '--gpu', 
        action='store_true', 
        help='Use GPU if available'
    )

    return parser.parse_args()