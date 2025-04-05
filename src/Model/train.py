import torch.optim as optim
import torch.nn as nn
import torch

def train(model, dataloader, epochs=10, lr=1e-4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for layers, full_image in dataloader:
            layers, full_image = layers.to(device), full_image.to(device)
            optimizer.zero_grad()
            output = model(layers)
            loss = criterion(output, full_image) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
