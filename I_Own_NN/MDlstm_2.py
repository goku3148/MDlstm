import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)

        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)

        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)

        self.W_io = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x, init_states=None):
        h_t, c_t = (
            torch.zeros(self.hidden_size).to(device),
            torch.zeros(self.hidden_size).to(device),
        ) if init_states is None else init_states

        i_t = torch.sigmoid(self.W_ii(x) + self.b_ii + self.W_hi(h_t) + self.b_hi)
        f_t = torch.sigmoid(self.W_if(x) + self.b_if + self.W_hf(h_t) + self.b_hf)
        g_t = torch.tanh(self.W_ig(x) + self.b_ig + self.W_hg(h_t) + self.b_hg)
        o_t = torch.sigmoid(self.W_io(x) + self.b_io + self.W_ho(h_t) + self.b_ho)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class Multi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, nhidden_layer):
        super(Multi_LSTM, self).__init__()

        self.n_layer = nhidden_layer
        self.fLSTM = LSTMCell(input_size, hidden_size)
        self.layer = nn.ModuleList([LSTMCell(hidden_size, hidden_size) for _ in range(nhidden_layer)])

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
    def __init__(self, input_size, hidden_size, n_hidden_layer, n_axis, zhidden_size, seq_len, output_size):
        super(D3MCELL, self).__init__()

        self.n_axis = n_axis
        self.n_layer = n_hidden_layer
        self.zhs = zhidden_size
        self.hidden_size = hidden_size
        self.MLSTM = nn.ModuleList([Multi_LSTM(input_size, hidden_size, self.n_layer) for _ in range(self.n_axis)])

        self.inp_w = self.weights_conf()
        self.init_weights()
        self.linear = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(seq_len)])
        self.single_li = nn.Linear(hidden_size, 1).to(device)

    def weights_conf(self):
        ML_if = []
        for i in range(self.n_layer + 1):
            input_forgot_gate = []
            for _ in range(self.n_axis):
                W_ii = nn.Linear(self.hidden_size, self.zhs).to(device)
                W_hi = nn.Linear(self.zhs, self.zhs).to(device)
                b_ii = nn.Parameter(torch.Tensor(self.zhs)).to(device)
                b_hi = nn.Parameter(torch.Tensor(self.zhs)).to(device)

                W_if = nn.Linear(self.hidden_size, self.zhs).to(device)
                W_hf = nn.Linear(self.zhs, self.zhs).to(device)
                b_if = nn.Parameter(torch.Tensor(self.zhs)).to(device)
                b_hf = nn.Parameter(torch.Tensor(self.zhs)).to(device)

                W_ic = nn.Linear(self.hidden_size, self.zhs).to(device)
                W_hc = nn.Linear(self.zhs, self.zhs).to(device)
                b_ic = nn.Parameter(torch.Tensor(self.zhs)).to(device)
                b_hc = nn.Parameter(torch.Tensor(self.zhs)).to(device)

                W_ilc = nn.Linear(self.zhs, self.zhs).to(device)
                b_ilc = nn.Parameter(torch.Tensor(self.zhs)).to(device)

                W_ilh = nn.Linear(self.hidden_size, self.zhs).to(device)
                b_ilh = nn.Parameter(torch.Tensor(self.zhs)).to(device)

                pp = [W_ii, W_hi, b_ii, b_hi, W_if, W_hf, b_if, b_hf, W_ic, W_hc, b_ic, b_hc, W_ilc, b_ilc, W_ilh,
                      b_ilh]
                input_forgot_gate.append(pp)

            ML_if.append(input_forgot_gate)

        return ML_if

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def cell_fn(self, ps, h_t, aweights):
        h_t = h_t
        inpw = aweights
        pht, pct = ps
        su_h, su_c = torch.zeros(self.zhs).to(device), torch.zeros(self.zhs).to(device)

        for li in range(self.n_axis):
            ph_t = pht[li]
            pc_t = pct[li]
            inp_w = inpw[li]

            i_t = torch.sigmoid(
                self.MLSTM[li].fLSTM.W_ii(h_t) + self.MLSTM[li].fLSTM.b_ii + self.MLSTM[li].fLSTM.W_hi(ph_t) +
                self.MLSTM[li].fLSTM.b_hi)
            f_t = torch.sigmoid(
                self.MLSTM[li].fLSTM.W_if(h_t) + self.MLSTM[li].fLSTM.b_if + self.MLSTM[li].fLSTM.W_hf(ph_t) +
                self.MLSTM[li].fLSTM.b_hf)
            g_t = torch.tanh(
                self.MLSTM[li].fLSTM.W_ig(h_t) + self.MLSTM[li].fLSTM.b_ig + self.MLSTM[li].fLSTM.W_hg(ph_t) +
                self.MLSTM[li].fLSTM.b_hg)

            i_cell = i_t * pc_t
            c_cell = f_t * g_t + i_cell

            Wt_1, bt_1 = inp_w[12].to(device), inp_w[13].to(device)
            t_1 = i_cell @ Wt_1 + bt_1

            Wt_2, bt_2 = inp_w[14].to(device), inp_w[15].to(device)
            t_2 = c_cell @ Wt_2 + bt_2

            su_h = su_h + t_2
            su_c = su_c + t_1

        mc_s = torch.sigmoid(su_h)
        mc_t = torch.softmax(su_c.reshape(1, 12), 1)[0]

        su_h_t = mc_s * mc_t

        return su_h_t

    def forward(self, x, init_states=None):
        fst = [[torch.zeros(self.hidden_size).to(device) for _ in range(2)]]
        output_, h_t, mlst = [torch.zeros(1).to(device), torch.zeros(self.hidden_size).to(device),
                              [fst + [[torch.zeros(self.hidden_size).to(device) for _ in range(2)] for _ in
                                      range(self.n_layer)] for _ in range(self.n_axis)]] \
            if init_states is None else init_states

        seq_le, input_size = x.size()
        inp_w = self.inp_w
        output_seq = torch.tensor([]).to(device)

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

        return output_seq, h_t, mlst
