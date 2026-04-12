import sys
import os

# Add src to path so imports work correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import device
from src.model import buildModel
from src.dataLoad import (
    loadWM38,
    getWM38Loaders,
    loadWM811K,
    preprocessWM811K,
    getWM811KLoaders
)
from src.training import trainCNN, trainWM811K
from src.accuracy import evaluateCNN, countDefectsPerClass
from src.GAN import getClassLoaders, trainGANs, visualizeGenerated

def main():
    print(f"Using device: {device}")
    print("="*50)

    # -------------------------------------------------------------------
    # MixedWM38 Pipeline
    # -------------------------------------------------------------------
    print("\nLoading MixedWM38 dataset...")
    images, labels = loadWM38()

    print("\nCreating WM38 dataloaders...")
    train_loader, test_loader, X_train, X_test, y_train, y_test = getWM38Loaders(images, labels)

    print("\nBuilding CNN model...")
    model, criterion_class, optimizer = buildModel()

    print("\nTraining CNN on MixedWM38...")
    trainCNN(model, train_loader, optimizer, criterion_class)

    print("\nEvaluating CNN on MixedWM38...")
    evaluateCNN(model, test_loader)

    print("\nCounting defects per class...")
    countDefectsPerClass(images, labels)

    # -------------------------------------------------------------------
    # WM-811K Pipeline
    # -------------------------------------------------------------------
    print("\nLoading WM-811K dataset...")
    wm811K = loadWM811K()

    print("\nPreprocessing WM-811K...")
    wm_images, wm_labels = preprocessWM811K(wm811K)

    print("\nCreating WM-811K dataloaders...")
    wm_train_loader, wm_test_loader, wm_X_train, wm_X_test, wm_y_train, wm_y_test = getWM811KLoaders(wm_images, wm_labels)

    print("\nBuilding WM-811K CNN model...")
    wm_model, wm_criterion, wm_optimizer = buildModel()

    print("\nTraining CNN on WM-811K...")
    trainWM811K(wm_model, wm_train_loader, wm_optimizer, wm_criterion)

    print("\nEvaluating CNN on WM-811K...")
    evaluateCNN(wm_model, wm_test_loader)

    # -------------------------------------------------------------------
    # GAN Pipeline
    # -------------------------------------------------------------------
    print("\nPreparing GAN class loaders...")
    class_loaders = getClassLoaders(images, labels)

    print("\nTraining GANs for minority classes...")
    trained_gans = trainGANs(class_loaders)

    print("\nVisualizing generated images...")
    visualizeGenerated(trained_gans)

    print("\nDone!")

if __name__ == "__main__":
    main()