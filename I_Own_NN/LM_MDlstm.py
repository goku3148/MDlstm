import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTMCell, self).__init__()
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

    def forward(self, x, init_states=None):
        """
        Forward pass for a single time step.

        Args:
            x (torch.Tensor): Input tensor for a single time step.
            init_states (tuple): Initial hidden and cell states. If None, initializes to zeros.

        Returns:
            tuple: Updated hidden and cell states.
        """

        h_t, c_t = (torch.zeros(self.hidden_size).to(self.self.device), torch.zeros(self.hidden_size).to(self.device)) \
            if init_states is None else init_states

        # Input gate
        i_t = torch.sigmoid(x @ self.W_ii.t() + self.b_ii + h_t @ self.W_hi.t() + self.b_hi)

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


class Multi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, nhidden_layer,device):
        super(Multi_LSTM, self).__init__()

        self.n_layer = nhidden_layer
        self.fLSTM = LSTMCell(input_size, hidden_size, device)
        self.layer = nn.ModuleList([LSTMCell(hidden_size, hidden_size, device) for _ in range(nhidden_layer)])

    def forward(self, x, init_states=None):
        lst = init_states

        oh_t, oc_t = [], []

        h_t0, c_t0 = lst[0]
        h_tl, c_tl = self.fLSTM(x, (h_t0, c_t0))
        lst[0] = [h_tl, c_tl]

        oh_t.append(h_tl)
        oc_t.append(c_tl)

        for i, layer in enumerate(self.layer):
            h_t1, c_t1 = lst[i + 1]
            h_tl, c_tl = layer(h_tl, (h_t1, c_t1))
            lst[i + 1] = h_tl, c_tl

            oh_t.append(h_tl)
            oc_t.append(c_tl)

        return oh_t, oc_t, lst


class D3MCELL(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden_layer, n_axis, zhidden_size, seq_len, output_size, device, em_size):
        super(D3MCELL, self).__init__()

        self.n_axis = n_axis
        self.n_layer = n_hidden_layer
        self.zhs = zhidden_size
        self.hidden_size = hidden_size
        self.MLSTM = nn.ModuleList([Multi_LSTM(input_size, hidden_size, self.n_layer, device) for _ in range(self.n_axis)])
        self.device = device
        self.inp_w = self.weights_conf()
        self.init_weights()
        self.linear = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(seq_len)])
        self.single_li = nn.Linear(em_size, hidden_size)

    def weights_conf(self):

        ML_if, MLs_p = [], []
        for i in range(self.n_layer + 1):
            input_forgot_gate = []
            # these are branches parameters
            for _ in range(self.n_axis):
                W_ii = nn.Parameter(torch.Tensor(self.zhs, self.hidden_size))
                W_hi = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
                b_ii = nn.Parameter(torch.Tensor(self.zhs))
                b_hi = nn.Parameter(torch.Tensor(self.zhs))

                W_if = nn.Parameter(torch.Tensor(self.zhs, self.hidden_size))
                W_hf = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
                b_if = nn.Parameter(torch.Tensor(self.zhs))
                b_hf = nn.Parameter(torch.Tensor(self.zhs))

                W_ic = nn.Parameter(torch.Tensor(self.zhs, self.hidden_size))
                W_hc = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
                b_ic = nn.Parameter(torch.Tensor(self.zhs))
                b_hc = nn.Parameter(torch.Tensor(self.zhs))

                W_ilc = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
                b_ilc = nn.Parameter(torch.Tensor(self.zhs))

                W_ilh = nn.Parameter(torch.Tensor(self.zhs, self.hidden_size))
                b_ilh = nn.Parameter(torch.Tensor(self.zhs))

                pp = [W_ii, W_hi, b_ii, b_hi, W_if, W_hf, b_if, b_hf, W_ic, W_hc, b_ic, b_hc, W_ilc, b_ilc, W_ilh,
                      b_ilh]
                input_forgot_gate.append(pp)

            ML_if.append(input_forgot_gate)

            # these are super parameters
            sW_ilc = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
            sb_ilc = nn.Parameter(torch.Tensor(self.zhs))

            sW_ilh = nn.Parameter(torch.Tensor(self.zhs, self.hidden_size))
            sb_ilh = nn.Parameter(torch.Tensor(self.zhs))

            sW_ilc = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
            sb_ilc = nn.Parameter(torch.Tensor(self.zhs))

            sW_ilh = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
            sb_ilh = nn.Parameter(torch.Tensor(self.zhs))

            W_io = nn.Parameter(torch.Tensor(self.zhs, self.hidden_size))
            W_ho = nn.Parameter(torch.Tensor(self.zhs, self.zhs))
            b_io = nn.Parameter(torch.Tensor(self.zhs))
            b_ho = nn.Parameter(torch.Tensor(self.zhs))

            s_p = [sW_ilc, sW_ilh, sb_ilc, sb_ilh, W_io, W_ho, b_io, b_ho]

            MLs_p.append(s_p)

        return ML_if

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

        for hl in range(self.n_layer):
            for params in self.inp_w[hl]:
                for p in params:
                    if p.data.ndimension() >= 2:
                        nn.init.xavier_uniform_(p.data)
                    else:
                        nn.init.zeros_(p.data)

        """for hl in range(self.n_layer):
            for p in self.sp_w[hl]:
                if p.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(p.data)
                else:
                    nn.init.zeros_(p.data)"""

    def cell_fn(self, ps, h_t, aweights):

        h_t = h_t
        inpw = aweights

        pht, pct = ps
        su_h, su_c = torch.zeros(self.zhs).to(self.device), torch.zeros(self.zhs).to(self.device)
        for li in range(self.n_axis):
            ph_t = pht[li]
            pc_t = pct[li]
            inp_w = inpw[li]

            W_ii, b_ii = inp_w[0].to(self.device), inp_w[2].to(self.device)
            W_hi, b_hi = inp_w[1].to(self.device), inp_w[3].to(self.device)
            W_if, b_if = inp_w[4].to(self.device), inp_w[6].to(self.device)
            W_hf, b_hf = inp_w[5].to(self.device), inp_w[7].to(self.device)
            W_ic, b_ic = inp_w[8].to(self.device), inp_w[10].to(self.device)
            W_hc, b_hc = inp_w[9].to(self.device), inp_w[11].to(self.device)

            i_t = torch.sigmoid(h_t @ W_ii.t() + b_ii + ph_t @ W_hi.t() + b_hi)

            f_t = torch.sigmoid(h_t @ W_if.t() + b_if + ph_t @ W_hf.t() + b_hf)

            g_t = torch.tanh(h_t @ W_ic.t() + b_ic + ph_t @ W_hc.t() + b_hc)

            i_cell = i_t * pc_t
            c_cell = f_t * g_t + i_cell

            Wt_1, bt_1 = inp_w[12].to(self.device), inp_w[13].to(self.device)
            t_1 = i_cell @ Wt_1 + bt_1

            Wt_2, bt_2 = inp_w[14].to(self.device), inp_w[15].to(self.device)
            t_2 = c_cell @ Wt_1 + bt_1

            su_h = su_h + t_2
            su_c = su_c + t_1

        mc_s = torch.sigmoid(su_h)
        mc_t = torch.softmax(su_c.reshape(1, 12), 1)[0]

        su_h_t = mc_s * mc_t

        return su_h_t

    def forward(self, x, init_states=None):

        fst = [[torch.zeros(self.hidden_size).to(self.device) for _ in range(2)]]
        output_, h_t, mlst = [torch.zeros(1).to(self.device), torch.zeros(self.hidden_size).to(self.device), [
            fst + [[torch.zeros(self.hidden_size).to(self.device) for _ in range(2)] for _ in range(self.n_layer)] for _ in
            range(self.n_axis)]] \
            if init_states is None else init_states

        seq_le, input_size = x.size()
        inp_w = self.inp_w
        output_seq = torch.tensor([]).to(self.device)
        for sl, linear in enumerate(self.linear):
            ls = 0
            axis_data = [[], []]
            for axis in self.MLSTM:
                In = x[sl]
                sh_t, sc_t, lst = axis(In, mlst[ls])
                axis_data[0].append(sh_t)
                axis_data[1].append(sc_t)
                mlst[ls] = lst
                ls += 1
            n_ad = []
            for i in range(self.n_layer + 1):
                x_ax, y_ax = [], []
                for j in range(self.n_axis):
                    x_ax.append(axis_data[0][j][i])
                    y_ax.append(axis_data[1][j][i])
                n_ad.append([x_ax, y_ax])
            for n_l in range(self.n_layer + 1):
                h_t = self.cell_fn(n_ad[n_l], h_t, inp_w[n_l])
                if sl == seq_le - 1:
                    d1ot = linear(h_t)
                    output_seq = torch.cat((output_seq, d1ot), dim=0)

        # output_ = self.single_li(h_t)

        return output_seq, h_t, mlst