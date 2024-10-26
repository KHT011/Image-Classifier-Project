import torch

def save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, epochs, class_to_idx):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'class_to_idx': class_to_idx 
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')