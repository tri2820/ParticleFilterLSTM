import torch
from torch.utils import data
from tt_dataset import TTDataset
from vanilla_models import VanillaLSTM


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 128,
          'shuffle': False,
          'num_workers': 1}
max_epochs = 5

# Generators
training_set = TTDataset(
          "data/train-data-measurements.npy",
          "data/train-data-ground_truth.npy",
          transform=True,
          remove_dummy=True
          )
training_generator = data.DataLoader(training_set, **params)

validation_set = TTDataset(
    "data/test-data-ca-measurements.npy",
    "data/test-data-ca-ground_truth.npy",
    transform=training_set.transformer,
    remove_dummy=True
    )
validation_generator = data.DataLoader(validation_set, **params)


model_params = {
    'batch_first':True, 
    'dropout':0.2,
    'num_layers':1
}
model = VanillaLSTM(input_dim=1, hidden_dim=32, target_dim=1, **model_params).double()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for epoch in range(max_epochs):
    # Training
    # 'Length' of training generator = number of samples / batch size
    for step, (local_batch, local_labels) in enumerate(training_generator,0):
        model.zero_grad()
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # local_batch shape = (batch_size, seq_size, input_size)
        # -> LSTM module in pytorch must set batch_first=True
        
        output,_ = model(local_batch)
        loss = criterion(output, local_labels)
        loss.backward()

        # Uncomment to check gradient flow
        # print([(p[0],p[1].grad) for p in list(model.named_parameters())])

        optimizer.step()
        
        if step%100==0:
            print(f"iteration {(epoch, step)}:\t{loss.item()}")


    # Validation
    with torch.set_grad_enabled(False):
        for step, (local_batch, local_labels) in enumerate(validation_generator,0):
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            output,_ = model(local_batch)
            loss = criterion(output, local_labels)
            print(f"validation, iteration {(epoch,step)}:\t{loss.item()}")


# Save model
import time
saving_path = f"models/model_vanilla_{int(time.time())}"
print(f"Saving model at: {saving_path}")
torch.save(model.state_dict(), saving_path)