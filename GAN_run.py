import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import device
from src.dataLoad import loadWM38
from src.GAN import getClassLoaders, trainGANs, visualizeGenerated

def main():
    print(f"Using device: {device}")

    print("\nLoading MixedWM38 dataset...")
    images, labels = loadWM38()

    print("\nPreparing per class dataloaders...")
    class_loaders = getClassLoaders(images, labels)

    print("\nTraining GANs for minority classes...")
    trained_gans = trainGANs(class_loaders)

    print("\nVisualizing generated images...")
    visualizeGenerated(trained_gans)

if __name__ == "__main__":
    main()