import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from get_input_args_predict import get_input_args
from load_checkpoint import load_checkpoint
from process_image import process_image
from load_category_names import load_category_names

def main():

    # Get command line arguments
    in_args = get_input_args()

    print("Loading model from " + in_args.checkpoint)

    # Load checkpoint
    model = load_checkpoint(in_args.checkpoint)

    print("Model loaded.")

    # Set device
    device = torch.device('cuda' if in_args.gpu and torch.cuda.is_available() else 'cpu')

    # Load the image
    img = Image.open(in_args.image)

    # Preprocess the image
    img = process_image(img)

    # Change device and add dimension to be 4D
    img = img.unsqueeze(0).to(device)

    # Map index to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Predict
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output)

    top_p, top_class = ps.topk(in_args.top_k, dim=1)

    # Convert to list
    top_p = top_p.flatten().tolist()
    top_class = [idx_to_class[i] for i in top_class.flatten().tolist()]

    # If a category names file is provided, map the class names to real names
    if in_args.category_names:
        cat_to_name = load_category_names(in_args.category_names)
        top_class = [cat_to_name[c] for c in top_class]

    # Print out the top K classes and their corresponding probabilities
    print(f"\nTop {in_args.top_k} Predictions:")
    for i in range(in_args.top_k):
        print(f"{top_class[i]}: {top_p[i]:.4f}")
    print()


if __name__ == '__main__':
    main()