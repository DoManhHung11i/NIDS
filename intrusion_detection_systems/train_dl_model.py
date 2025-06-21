import torch
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                if inputs_val.dim() == 2:
                    inputs_val = inputs_val.unsqueeze(-1)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item() * inputs_val.size(0)
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    print(f"Finished Training {model.__class__.__name__}")
    return model

def evaluate_model(model, data_loader, device, model_type=None, name="Test"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if model_type in ['cnn', 'rnn'] and inputs.dim() == 2:
                inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n---------- {name} data ({model.__class__.__name__})")
    print(classification_report(all_labels, all_preds, zero_division=0))
    print("Accuracy: ", accuracy_score(all_labels, all_preds))
    print("Detection Rate (Recall): ", recall_score(all_labels, all_preds, pos_label=1))
    print("F1 Score: ", f1_score(all_labels, all_preds, pos_label=1))
    try:
        print("ROC AUC Score: ", roc_auc_score(all_labels, all_preds))
    except ValueError as e:
        print(f"ROC AUC Score: Not applicable or error - {e}")
