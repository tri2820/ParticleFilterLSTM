"""
Consits of 2 <b>separated</b> LSTM
Since there are 2 seperated task in target tracking
1. Estimating the real state aka filtering
2. Predicting the next state aka prediction

One could train these 2 networks seperatedly,
however for the sake of keeping the `target tracking` spirit
we will train them together. This is a rough simulation for realworld
realtime problems where new information come once and 
we instantaneously update the networks together.

The implemetation could be done 
due to the fact that NN are data-driven,
in contrast to Bayesian recursive idea.
"""
import torch
from vanilla_models import VanillaLSTM

class NBLSTM(torch.nn.Module):
    def __init__(self, z_dim, x_dim, filtering_params=None, prediction_params=None):
        super(NBLSTM, self).__init__()
        
        # Default is params of SNBLSTM 
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

        # VanillaLSTM explains the absence of the linear mapping input->hidden
        self.filtering_model = VanillaLSTM(input_dim=z_dim,target_dim=x_dim, **self.filtering_params)

        # VanillaLSTM explains the absence of the linear mapping input->hidden
        self.prediction_model = VanillaLSTM(input_dim=z_dim,target_dim=x_dim, **self.prediction_params)
        

    def forward(self, measurements):
        filtering_outputs,_ = self.filtering_model(measurements)
        prediction_outputs,_ = self.filtering_model(measurements)

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

    model = NBLSTM(1,1).double()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


    for epoch in range(max_epochs):
        # Training
        for step, (local_batch, local_labels) in enumerate(training_generator,0):
            model.zero_grad()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            filtering_output, prediction_output  = model(local_batch)

            loss_f = criterion(filtering_output, local_labels)
            loss_f.backward()

            # Shift to calculate loss on prediction module
            # Omit the last state in each series of prediction output
            # since we do not have the truthground of the next state of the last state (in this batch)
            # Omit the first state in each series of local labels
            # since we do not predict it
            loss_p = criterion(prediction_output[:,:-1,:], local_labels[:,1:,:])
            loss_p.backward()

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
    saving_path = f"models/model_NBLSTM_{int(time.time())}"
    print(f"Saving model at: {saving_path}")
    torch.save(model.state_dict(), saving_path)