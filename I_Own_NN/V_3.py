import torch
import torch.nn as nn
import numpy as np

device = torch.device('cpu')
torch.set_default_dtype(torch.float32)

class Branch_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, ahidden_size, naxis, device):
        super(Branch_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ahidden_size = ahidden_size
        self.naxis = naxis
        self.device = device
        self.axis_w,self.db_w = self.weights_conf()
        self.init_weights(self.axis_w,self.db_w)



    def weights_conf(self):

        axis_w = []
        # Unit_cell_parameters
        for i in range(self.naxis):

            W_ii = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size)).to(self.device)
            W_hi = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)).to(self.device)
            input_gate = [W_ii,W_hi]

            W_if = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size)).to(self.device)
            W_hf = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)).to(self.device)
            forgot_gate = [W_if, W_hf]

            W_ig = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size)).to(self.device)
            W_hg = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)).to(self.device)
            call_gate = [W_ig, W_hg]

            W_io = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size)).to(self.device)
            W_ho = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)).to(self.device)
            output_gate = [W_io,W_ho]

            axis_w.append([input_gate,forgot_gate,call_gate,output_gate])

        # DB_compiler_layer
        W_ii = nn.Parameter(torch.Tensor(self.naxis, self.hidden_size, self.ahidden_size)).to(self.device)
        W_hi = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size, self.ahidden_size)).to(self.device)
        b_ii = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        b_hi = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        input_gate = [W_ii, W_hi, b_ii, b_hi]

        W_if = nn.Parameter(torch.Tensor(self.naxis, self.hidden_size, self.ahidden_size)).to(self.device)
        W_hf = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size, self.ahidden_size)).to(self.device)
        b_if = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        b_hf = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        forgot_gate = [W_if, W_hf, b_if, b_hf]

        W_ig = nn.Parameter(torch.Tensor(self.naxis, self.hidden_size, self.ahidden_size)).to(self.device)
        W_hg = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size, self.ahidden_size)).to(self.device)
        b_ig = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        b_hg = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        call_gate = [W_ig, W_hg, b_ig, b_hg]

        W_ilc = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size, self.ahidden_size)).to(self.device)
        b_ilc = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        ot_c = [W_ilc, b_ilc]

        W_ilh = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size, self.ahidden_size)).to(self.device)
        b_ilh = nn.Parameter(torch.Tensor(self.naxis, self.ahidden_size)).to(self.device)
        W_ict = nn.Parameter(torch.Tensor(self.naxis, self.hidden_size, self.ahidden_size)).to(self.device)
        ot_h = [W_ilh, b_ilh,W_ict]



        DB = [input_gate,forgot_gate,call_gate,ot_c,ot_h]

        return axis_w,DB
    def init_weights(self,weight_conf,weight_db):
        unit_cell = weight_conf
        for ax in unit_cell:
            for gt in ax:
                for p in gt:
                    if p.data.ndimension() >= 2:
                        nn.init.xavier_uniform_(p.data)
                    else:
                        nn.init.zeros_(p.data)
        db = weight_db
        for ax in db:
            for p in ax:
                if p.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(p.data)
                else:
                    nn.init.zeros_(p.data)


    def diff_percent(self, x, h):
        pv = (x-h)/x
        return 1 - torch.abs(pv)

    def forward(self,x,branch_state=None,db_state=None):

        seq,ft = x.shape
        h_t,c_t = torch.zeros(self.naxis, seq, self.hidden_size).to(self.device), torch.zeros(self.naxis ,seq,self.hidden_size).to(self.device)  \
             if branch_state is None else branch_state
        axis_w = self.axis_w
        h_m,c_m = torch.tensor([]),torch.tensor([])
        for i in range(self.naxis):
            h_x,c_x = branch_state
            h_x,c_x = h_x[i],c_x[i]
            weights = axis_w[i]
            W_if,W_hf = weights[1][0],weights[1][1]
            W_ii,W_hi = weights[0][0],weights[0][1]
            W_ig,W_hg = weights[2][0],weights[2][1]
            W_io,W_ho = weights[3][0],weights[3][1]

            fg_p = self.diff_percent(x@W_if, h_x@W_hf)

            ig_p = torch.sigmoid(x@W_ii + h_x@W_hi)

            gg_t = torch.tanh(x@W_ig + h_x@W_hg)

            og_t = torch.sigmoid(x@W_io + h_x@W_ho)

            c_t1 =  fg_p * c_x + ig_p*gg_t
            h_t1 = og_t * torch.tanh(c_t1)

            h_m = torch.cat((h_m,h_t1.reshape(1,seq,self.hidden_size)),dim=0)
            c_m = torch.cat((c_m,c_t1.reshape(1,seq,self.hidden_size)),dim=0)


        h_db = torch.zeros(seq,self.ahidden_size) \
             if db_state is None else db_state

        inp_g = self.db_w[0]
        fnp_g = self.db_w[1]
        cng_g = self.db_w[2]
        ocg_g = self.db_w[3]
        ohg_g = self.db_w[4]

        su_h, su_c = torch.zeros(self.naxis,seq,self.ahidden_size).to(self.device), torch.zeros(self.naxis,seq,self.ahidden_size).to(self.device)
        for i in range(self.naxis):

            W_idb, W_hdb, b_idb, b_hdb = inp_g[0][i], inp_g[1][i], inp_g[2][i], inp_g[3][i]
            W_idbf, W_hdbf, b_idbf, b_hdbf = fnp_g[0][i], fnp_g[1][i], fnp_g[2][i], fnp_g[3][i]
            W_idbc, W_hdbc, b_idbc, b_hdbc = cng_g[0][i], cng_g[1][i], cng_g[2][i], cng_g[3][i]


            h_dx = h_db
            h_t = h_m[i]
            c_t = c_m[i] @ ohg_g[2][i]

            in_db = torch.sigmoid(h_t @ W_idb + h_dx @ W_hdb + b_idb + b_hdb)

            fn_db = torch.sigmoid(h_t @ W_idbf + h_dx @ W_hdbf + b_idbf + b_hdbf)

            cn_db = torch.tanh(h_t @ W_idbc + h_dx @ W_hdbc + b_idbc + b_hdbc)

            i_cell = in_db * c_t
            c_cell = fn_db * cn_db + i_cell

            Wt_1, bt_1 = ocg_g[0][i],ocg_g[1][i]
            t_1 = i_cell @ Wt_1 + bt_1

            Wt_2, bt_2 = ohg_g[0][i],ohg_g[1][i]
            t_2 = c_cell @ Wt_2 + bt_2

            su_h[i] = su_h[i] + t_2
            su_c[i] = su_c[i] + t_1

        mc_s = torch.sigmoid(su_h)
        mc_t = torch.tanh(su_c)

        h_db = mc_s * mc_t


        return (h_m,c_m),h_db


class V_2(nn.Module):
    def __init__(self, input_size, hidden_size, ahidden_size, naxis, layers, device):
        super(V_2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ahidden_size = ahidden_size
        self.naxis = naxis
        self.layers = layers
        self.device = device
        self.b_cell = nn.ModuleList([Branch_Cell(input_size,hidden_size,ahidden_size,naxis,device) for _ in range(self.layers)])

    def forward(self,x,branch_state=None,db_state=None):

        seq,feat = x.shape

        h_t, c_t = torch.zeros(self.naxis, seq, self.hidden_size).to(self.device), torch.zeros(self.naxis, seq,self.hidden_size).to(self.device) \
            if branch_state is None else branch_state

        h_db = torch.zeros(seq, self.ahidden_size) \
            if db_state is None else db_state

        for layer in self.b_cell:
            hcm,h_db = layer(x=x,branch_state=(h_t,c_t),db_state=h_db)
        print(h_db.shape)
        return h_db






model = V_2(5,10,20,4,5,device)
def diff_percent(x, h):
        return 1 - torch.abs((x - h) / x)

X = torch.rand(30,5,dtype=torch.float32)
print(model(X))


