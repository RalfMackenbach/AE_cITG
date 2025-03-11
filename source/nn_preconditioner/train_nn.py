# Here we train the neural network
import os 
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import precon_nn as pnn
import h5py
import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import wandb


# set hyperparameters
batch_size = 2**13
learning_rate = 0.001
epochs = 1000
weight_decay = 1e-6
hidden_units = 32
hidden_layers = 10

# fix the seed
torch.manual_seed(0)
np.random.seed(0)


# Set the device to MPS
mps_device = torch.device("mps")

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# first we have to collect the data
path = '/Users/rjjm/Library/CloudStorage/ProtonDrive-ralf.mackenbach@proton.me-folder/ITG_data/AE_data/'
# get all hdf5 files
files = os.listdir(path)
files = [f for f in files if f.endswith('.hdf5')]
# sort the files
files.sort()
# only keep last file
files = [files[0]]

k_alpha = []
k_psi = []
w_alpha = []
w_psi = []
w_n = []
w_T = []

# loop over the files
for f in files:
    # load the data
    with h5py.File(path+'/'+f, 'r') as hf:
        # get the data
        data = hf
        # loop over the tubes
        for tube in tqdm.tqdm(data.keys(), desc=f'Processing {f}'):
            # get the data (arrays)
            k_alpha_arr = data[tube]['k_alpha_arr'][()]
            k_psi_arr = data[tube]['k_psi_arr'][()]
            w_alpha_arr = data[tube]['w_alpha'][()]
            w_psi_arr = data[tube]['w_psi'][()]
            w_n_arr = data[tube]['w_n'][()] * np.ones_like(k_alpha_arr)
            w_T_arr = data[tube]['w_T'][()] * np.ones_like(k_alpha_arr)
            # append to the list
            k_alpha.append(k_alpha_arr)
            k_psi.append(k_psi_arr)
            w_alpha.append(w_alpha_arr)
            w_psi.append(w_psi_arr)
            w_n.append(w_n_arr)
            w_T.append(w_T_arr)

# convert to 1D numpy arrays
k_alpha = np.concatenate(k_alpha)
k_psi = np.concatenate(k_psi)
w_alpha = np.concatenate(w_alpha)
w_psi = np.concatenate(w_psi)
w_n = np.concatenate(w_n)
w_T = np.concatenate(w_T)

print(f'Shape of arrays: {k_alpha.shape}')

# convert to torch tensors (Nx4) and (Nx2)
inputs = torch.tensor(np.vstack([w_n,w_T,w_alpha,w_psi]).T).float().to(mps_device)
targets = torch.tensor(np.vstack([k_alpha,k_psi]).T).float().to(mps_device)

# shuffle the data
indices = np.arange(inputs.shape[0])
np.random.shuffle(indices)
inputs = inputs[indices]
targets = targets[indices]
# split the data
N = inputs.shape[0]
N_train = int(0.8 * N)
N_test = N - N_train
inputs_train = inputs[:N_train]
targets_train = targets[:N_train]
inputs_test = inputs[N_train:]
targets_test = targets[N_train:]

# create datasets
train_dataset = CustomDataset(inputs_train, targets_train)
test_dataset = CustomDataset(inputs_test, targets_test)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)

# print info
print(f'Number of samples: {N}')
print(f'Number of training samples: {N_train}')
print(f'Number of test samples: {N_test}', end='\n\n')

# Initialize wandb
wandb.init(project="precon_nn_project")

# get the precon_nn model and training function
model = pnn.SimpleNN(4, hidden_units, hidden_layers, 2).to(mps_device)
# use custom loss
def criterion(outputs, targets):
    loss = nn.MSELoss()(outputs, targets)
    return loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Log the model configuration
wandb.config.update({
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "weight_decay": weight_decay,
    "hidden_units": hidden_units,
    "hidden_layers": hidden_layers,
})

# train the model
losses = []
for epoch in range(1000):
    model.train()
    for inputs_batch, targets_batch in train_loader:
        inputs_batch, targets_batch = inputs_batch.to(mps_device), targets_batch.to(mps_device)
        optimizer.zero_grad()
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()
    
    losses.append(loss.item())
    
    # Log the loss to wandb
    wandb.log({"epoch": epoch, "loss": loss.item()})

    if epoch % 1 == 0:
        print(f'Epoch {epoch}, log10(loss) = {np.log10(loss.item())}')

# plot the loss
plt.semilogy(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
        

# test the model
model.eval()
test_loss = 0.0
criterion_test = nn.L1Loss()
with torch.no_grad():
    for inputs_batch, targets_batch in test_loader:
        inputs_batch, targets_batch = inputs_batch.to(mps_device), targets_batch.to(mps_device)
        outputs = model(inputs_batch)
        loss = criterion_test(outputs, targets_batch)
        test_loss += loss.item()

test_loss /= len(test_loader)

# Log the test loss to wandb
wandb.log({"test_loss": test_loss})

# print the test loss
print(f'Test loss: {test_loss}')


# save the model
torch.save(model.state_dict(), 'precon_nn.pth')
print('Model saved as precon_nn.pth')

# Finish the wandb run
wandb.finish()