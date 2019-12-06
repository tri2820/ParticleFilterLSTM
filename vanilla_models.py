import torch
import torch.nn.functional as F

class VanillaLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim, **params):
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Since LSTM of Pytorch has a linear to map input_dim -> hidden_dim
        # The linear module for the same task described in the paper is implicitly implemented here
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, **params)
        self.linear = torch.nn.Linear(hidden_dim, target_dim)

    def forward(self, measurements, hidden_hc=None):
        batch_size = measurements.shape[0]

        outputs, hiden_hc = self.lstm(measurements, hidden_hc)
        # outputs is all the hidden states of the final layer through time 
        # outputs is considered as batch of batch of hidden state

        # (1) Alternative way to apply linear to all time steps:
        # output_states = [self.linear(batch) for batch in outputs]
        # output_states = torch.cat(output_states).view(batch_size,-1,1)

        # (2) For the sake of computational efficiency, 
        # we merge the first and second batch dimension to one
        # then adjust the size of output_states later
        output_states = self.linear(outputs.contiguous().view(-1,self.hidden_dim))
        output_states = output_states.view(batch_size,-1,1)     

        # Activation function is ommited since it's not mentioned in the paper   
        # output_states = F.leaky_relu(output_states)
        return output_states, hiden_hc


if __name__ == "__main__":
    torch.manual_seed(1)
    vlstm = VanillaLSTM(3,2,1)

    # batch of size 4, sequence of length 5, state at a timestep is a Tensor(x,y,z)
    inputs = torch.randn(4,5,3)
    outputs = vlstm(inputs)
    print(outputs)