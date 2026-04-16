import torch
import torch.nn as nn
from src.config import (
    CNN_EPOCHS,
    device
)

# -------------------------------------------------------------------
# CNN Training
# -------------------------------------------------------------------
def trainCNN(model, train_loader, optimizer, criterion_class):
    for epoch in range(CNN_EPOCHS):
        model.train()
        current_loss = 0.0
        correct_train = 0
        total_train = 0

        for image, labels in train_loader:
            image = image.to(device)
            labels = labels.to(device)

            Classification_out = model(image)
            loss = criterion_class(Classification_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

            # Training accuracy
            predicted = torch.sigmoid(Classification_out) > 0.5
            correct_train += (predicted == labels).all(dim=1).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        print(f"Epoch [{epoch+1}/{CNN_EPOCHS}], Loss: {current_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")


# -------------------------------------------------------------------
# WM-811K CNN Training
# -------------------------------------------------------------------
def trainWM811K(wm_model, wm_train_loader, wm_optimizer, wm_criterion):
    for epoch in range(CNN_EPOCHS):
        wm_model.train()
        current_loss = 0.0
        correct_train = 0
        total_train = 0

        for image, labels in wm_train_loader:
            image = image.to(device)
            labels = labels.to(device)

            outputs = wm_model(image)
            loss = wm_criterion(outputs, labels)

            wm_optimizer.zero_grad()
            loss.backward()
            wm_optimizer.step()

            current_loss += loss.item()

            predicted = torch.sigmoid(outputs) > 0.5
            correct_train += (predicted == labels).all(dim=1).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        print(f"Epoch [{epoch+1}/{CNN_EPOCHS}], Loss: {current_loss/len(wm_train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")