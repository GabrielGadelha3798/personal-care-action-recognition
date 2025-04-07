import os
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """Perform one training epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        batch_correct = torch.sum(preds == labels.data)
        batch_total = inputs.size(0)
        
        running_loss += loss.item() * batch_total
        running_corrects += batch_correct
        total_samples += batch_total
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(batch_correct / batch_total).item():.4f}'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = (running_corrects.double() / total_samples).cpu().numpy().item()
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """Perform validation on test set"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar: 
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            batch_correct = torch.sum(preds == labels.data)
            batch_total = inputs.size(0)
            
            running_loss += loss.item() * batch_total
            running_corrects += batch_correct
            total_samples += batch_total
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(batch_correct / batch_total).item():.4f}'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = (running_corrects.double() / total_samples).cpu().numpy().item()
    return epoch_loss, epoch_acc

def save_metrics(metrics, save_path):
    """Save training metrics to JSON file"""
    file_path = os.path.join(save_path, 'training_metrics.json')
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_metrics(metrics):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['test_loss'], label='Test Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                num_epochs=25, save_path='./model_checkpoints'):
    """Main training function with metrics logging"""
    os.makedirs(save_path, exist_ok=True)
    
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'best_test_acc': 0.0,
        'config': {
            'num_epochs': num_epochs,
            'optimizer': optimizer.__class__.__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'device': str(device)
        }
    }
    
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validation
        test_loss, test_acc = validate(
            model, test_loader, criterion, device, epoch, num_epochs
        )
        
        # Update metrics
        metrics['train_loss'].append(float(train_loss))
        metrics['train_acc'].append(float(train_acc))
        metrics['test_loss'].append(float(test_loss))
        metrics['test_acc'].append(float(test_acc))
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_path, f'model_e_{epoch+1}.pth'))
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            metrics['best_test_acc'] = float(best_test_acc)
        
        
        # Epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f'Test Loss:  {test_loss:.4f} | Acc: {test_acc:.4f}')
        print('-' * 60)
    
        # Save metrics and plot
        save_metrics(metrics, save_path)
        plot_metrics(metrics)
    
    return model, metrics