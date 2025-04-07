import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict

def evaluate_model(model, test_loader, device, class_names):
    """
    Valida o modelo em um conjunto de teste e gera relatórios de desempenho.

    Parâmetros:
    model (torch.nn.Module): O modelo treinado.
    test_loader (torch.utils.data.DataLoader): DataLoader para o conjunto de teste.
    device (torch.device): Dispositivo para computação (CPU ou GPU).
    class_names (list): Lista de nomes das classes.
    """
    model.eval()
    
    # Inicializa variáveis para armazenar resultados
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confidence_scores = []
    
    with torch.no_grad(): # Garante que não calculamos gradientes
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            max_probs, _ = torch.max(probabilities, 1)
            
            # Atualiza métricas
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Armazena previsões e rótulos
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confidence_scores.extend(max_probs.cpu().numpy())
            
            # Métricas por classe
            for lbl, pred, prob in zip(labels, predicted, max_probs):
                per_class_total[lbl.item()] += 1
                if lbl == pred:
                    per_class_correct[lbl.item()] += 1
    
    accuracy = 100 * correct / total
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Total Test Samples: {total}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Plot per-class accuracy
    class_accuracies = []
    for class_id in sorted(per_class_total.keys()):
        acc = 100 * per_class_correct[class_id] / per_class_total[class_id]
        class_accuracies.append(acc)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), class_accuracies, color='skyblue')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.title('Per-Class Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=20, color='purple', alpha=0.7)
    plt.title('Distribution of Prediction Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate and print class-wise metrics
    print("\nClass-wise Performance:")
    for class_id in sorted(per_class_total.keys()):
        acc = 100 * per_class_correct[class_id] / per_class_total[class_id]
        print(f"{class_names[class_id]:<15}: {per_class_correct[class_id]}/{per_class_total[class_id]} = {acc:.2f}%")
    
    return accuracy, cm