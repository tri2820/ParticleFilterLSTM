"""
BFLSTM keeps the idea of Bayesian recursive solution,
that is, the filtering and prediction interact with each others.

Solving target tracking problem is considered as only one main goal:
estimate the real trajectories that are close to the true tracks
aka minimizing the MSE 
between filtered states and predicted states and true states. (*)

The solution when we individually care for filtering 
or prediction can still be obtained by extract from corresponding NN.

Hyperparameter `e` is introduced
If `e`>1, accuracy filtering is more valued than that of prediction.

Footnote: for delta_t << 0, this is efficient (*)
"""

import torch
from vanilla_models import VanillaLSTM

class BFLSTM(torch.nn.Module):
    def __init__(self, z_dim, x_dim, filtering_params=None, prediction_params=None):
        super(BFLSTM, self).__init__()
        # VanillaLSTM explains the absence of the linear mapping input->hidden
         
        # Default is params of SBFLSTM 
        if filtering_params==None: 
            self.filtering_params = {
                'batch_first':True, 
                'dropout':0.2,
                'num_layers':1,
                'hidden_dim':32
            }
        else:
            self.filtering_params = filtering_params

        if prediction_params==None:
            self.prediction_params = {
                'batch_first':True, 
                'dropout':0.2,
                'num_layers':1,
                'hidden_dim':32
            }
        else:
            self.prediction_params = prediction_params
            
        self.filtering_model = VanillaLSTM(input_dim=z_dim,target_dim=x_dim, **self.filtering_params)
        # VanillaLSTM explains the absence of 
        # the linear mapping filtering_output->prediction_input
        self.prediction_model = VanillaLSTM(input_dim=x_dim,target_dim=x_dim, **self.prediction_params)
        

    def forward(self, measurements):
        filtering_outputs, hidden_hc = self.filtering_model(measurements)
        prediction_outputs, _ = self.filtering_model(filtering_outputs, hidden_hc)
        return filtering_outputs, prediction_outputs


if __name__ == "__main__":
    from torch.utils import data
    from tt_dataset import TTDataset
    
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



    model = BFLSTM(1,1).double()

    training_params = {'epsilon':1}
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


    for epoch in range(max_epochs):
        # Training
        for step, (local_batch, local_labels) in enumerate(training_generator,0):
            model.zero_grad()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            filtering_output, prediction_output  = model(local_batch)

            loss_f = criterion(filtering_output, local_labels)

            # Shift to calculate loss on prediction module
            # Omit the last state in each series of prediction output
            # since we do not have the truthground of the next state of the last state (in this batch)
            # Omit the first state in each series of local labels
            # since we do not predict it
            loss_p = criterion(prediction_output[:,:-1,:], local_labels[:,1:,:])

            loss = loss_f + training_params['epsilon']*loss_p
            loss.backward()

            optimizer.step()
            
            if step%100==0:
                print(f"iteration {(epoch, step)}:\t{(loss_f.item(),loss_p.item())}")


        # Validation
        with torch.set_grad_enabled(False):
            for step, (local_batch, local_labels) in enumerate(validation_generator,0):
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                filtering_output, prediction_output  = model(local_batch)
                loss_f = criterion(filtering_output, local_labels)
                loss_p = criterion(prediction_output[:,:-1,:], local_labels[:,1:,:])
                print(f"validation, iteration {(epoch,step)}:\t{(loss_f.item(),loss_p.item())}")


    # Save model
    import time
    saving_path = f"models/model_BFLSTM_{int(time.time())}"
    print(f"Saving model at: {saving_path}")
    torch.save(model.state_dict(), saving_path)