import torch.nn as nn
import torch

torch.optim.lr_scheduler.LambdaLR


class NLSTM_cell(nn.Module):
    def __init__(self,input_size, hidden_size, device):
        super(NLSTM_cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Input gate weights
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate weights
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate weights
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate weights
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def diff_percent(self, x, h):
        pv = (x-h)/x
        return 1 - torch.abs(pv)

    def forward(self, x, init_states=None):
        """
        Forward pass for a single time step.

        Args:
            x (torch.Tensor): Input tensor for a single time step.
            init_states (tuple): Initial hidden and cell states. If None, initializes to zeros.

        Returns:
            tuple: Updated hidden and cell states.
        """

        h_t, c_t = (torch.zeros(self.hidden_size).to(self.device), torch.zeros(self.hidden_size).to(self.device)) \
            if init_states is None else init_states

        # Input gate
        i_t = self.diff_percent(x @ self.W_ii.t() + self.b_ii,h_t @ self.W_hi.t() + self.b_hi)

        # Forget gate
        f_t = torch.sigmoid(x @ self.W_if.t() + self.b_if + h_t @ self.W_hf.t() + self.b_hf)

        # Cell gate
        g_t = torch.tanh(x @ self.W_ig.t() + self.b_ig + h_t @ self.W_hg.t() + self.b_hg)

        # Output gate
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + h_t @ self.W_ho.t() + self.b_ho)

        # Cell state update
        c_t = f_t * c_t + i_t * g_t

        # Hidden state update
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class FF(nn.Module):
    def __init__(self, input_size, hidden_size, device, output_size):
        super(FF, self).__init__()

        self.LSTM_layer = NLSTM_cell(input_size=input_size,hidden_size=hidden_size,device=device)
        self.Dense_layer = nn.Linear(hidden_size,output_size)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self,x):

        h_t, c_t = (torch.zeros(self.hidden_size).to(self.device), torch.zeros(self.hidden_size).to(self.device))
        seq,ft = x.shape

        for i in range(seq):
            h_t,c_t = self.LSTM_layer(x[i])

        output = self.Dense_layer(h_t)

        return output

device = torch.device('cpu')

model = FF(input_size=3,hidden_size=5,device=device,output_size=1)

X = torch.rand(30,3)

print(model(X))



