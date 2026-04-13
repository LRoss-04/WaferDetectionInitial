import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from .config import device, CLASS_NAMES

# -------------------------------------------------------------------
# CNN Evaluation
# -------------------------------------------------------------------
def evaluateCNN(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            class_out = model(images)
            predicted = torch.sigmoid(class_out) > 0.5

            # Exact match accuracy
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)

            all_predictions.append(predicted.cpu())
            all_targets.append(labels.cpu())

    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()

    accuracy = correct / total
    macro_f1 = f1_score(all_targets, all_predictions, average='macro')
    micro_f1 = f1_score(all_targets, all_predictions, average='micro')

    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print("\nPer Class Report:")
    print(classification_report(all_targets, all_predictions, target_names=CLASS_NAMES))

    return accuracy, macro_f1, micro_f1


# -------------------------------------------------------------------
# Defect Counter
# -------------------------------------------------------------------
def DefectCounter(image):
    # Counts defect dies after /2.0 normalization
    # defect dies are >= 1.0
    return (image >= 1.0).sum()


def countDefectsPerClass(images, labels):
    defect_array = np.zeros(9)

    for i in range(len(images)):
        imageNum = images[i]
        labelNum = labels[i]

        current_Count = DefectCounter(imageNum)

        for j in range(len(labelNum)):
            if labelNum[j] == 1:
                defect_array[j] += current_Count

    for j in range(9):
        print(f"{CLASS_NAMES[j]}: {int(defect_array[j])}")

    return defect_array