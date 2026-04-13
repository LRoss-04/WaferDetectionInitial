import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.config import device
from src.model import buildModel
from src.dataLoad import loadWM38, getWM38Loaders
from src.training import trainCNN
from src.accuracy import evaluateCNN, countDefectsPerClass

def main():
    print(f"Using device: {device}")

    print("\nLoading MixedWM38 dataset...")
    images, labels = loadWM38()

    print("\nCreating dataloaders...")
    train_loader, test_loader, X_train, X_test, y_train, y_test = getWM38Loaders(images, labels)

    print("\nBuilding model...")
    model, criterion_class, optimizer = buildModel()

    print("\nTraining...")
    trainCNN(model, train_loader, optimizer, criterion_class)

    print("\nEvaluating...")
    evaluateCNN(model, test_loader)

    print("\nCounting defects per class...")
    countDefectsPerClass(images, labels)

if __name__ == "__main__":
    main()