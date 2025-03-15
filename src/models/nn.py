import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PVNN(nn.Module):
    def __init__(self, input_size):
        super(PVNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(X_train, y_train, X_train_scaled, X_test, y_test, X_test_scaled):
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    batch_size = 342

    input_size = X_train_scaled.shape[1]
    model = PVNN(input_size)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
                        
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}, mean loss: {loss_sum/len(dataset)}") 

    # Test the model
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)    
    test_loss = criterion(model(X_test_tensor), y_test_tensor)
    print(f"Test loss: {test_loss.item()/len(X_test_tensor)}")

    return model