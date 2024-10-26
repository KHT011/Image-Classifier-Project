import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from get_input_args_train import get_input_args
from load_dataset import load_dataset
from create_model import create_model
from save_checkpoint import save_checkpoint

def main():

    # Get command line arguments
    in_args = get_input_args()

    # Load dataset and get class to index mapping
    dataloaders, class_to_idx = load_dataset(in_args.data_dir)

    print("\nDataset loaded.\n")

    # Set device
    device = torch.device('cuda' if in_args.gpu and torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_model(in_args.arch, in_args.hidden_units)

    print("Model created.\n")

    # Criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)

    # Train model
    model.to(device)

    print("Training started.\n")

    # Training loop
    for e in range(in_args.epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Validation and accuracy
        model.eval()
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model(inputs)
                loss = criterion(log_ps, labels)
                validation_loss += loss

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        model.train()

        # Print the validation and accuracy log
        print(f"Epoch {e+1}/{in_args.epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

    print("\nTraining completed.\n")

    # Save checkpoint
    save_checkpoint(model, in_args.save_dir, in_args.arch, in_args.hidden_units, in_args.learning_rate, in_args.epochs, class_to_idx)
    print("Checkpoint saved.\n")


if __name__ == '__main__':
    main()