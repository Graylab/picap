from torch import nn
import torch


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        #print("EdgeModel Feat:",source.shape,target.shape,radial.shape,edge_attr.shape)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index

        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        #print("Init:",h.shape,edge_index.shape,coord.shape)
        edge_index = edge_index.long()

        row, col = edge_index

        radial, coord_diff = self.coord2radial(edge_index, coord)

        #print("PostProcess:",h[row].shape,h[col].shape,radial.shape,edge_attr.shape)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.ReLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))

        self.to(self.device)
        self.norm = nn.BatchNorm1d(out_node_nf)

    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        h1 = h
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)


        h = h+h1
        h = self.embedding_out(h)
        # h=self.norm(h)



        h = torch.sigmoid(h)
        return h, x


class TYPE_BINDER(nn.Module):
    def __init__(self, in_node_nf=1280, hidden_nf=[128,128,128,128], out_node_nf=32, in_edge_nf=23,
            output_dim=3, dropout_rate=0.5, num_FcLayers=16, num_Fc_times=8,
            device='cpu', act_fn=nn.ReLU(), n_layers=[4,4,4,4],
            residual=True, attention=True, normalize=True, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features (array in this case of $4$)
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param output_dim: Number of output dimensions
        :param dropout_rate: Dropout rate post-graph
        :param num_drop_FcLayers: Number of FC layers post-graph
        :param num_Fc_times: Number of times each FC layers goes thru post-graph data
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(TYPE_BINDER, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding_in0 = nn.Linear(in_node_nf, self.hidden_nf[0])
        self.embedding_out = nn.Linear(self.hidden_nf[-1], out_node_nf)
        self.embedding_res = nn.Linear(in_node_nf, out_node_nf)
        self.graph_module = nn.ModuleList()

        self.graph_norm_end = nn.BatchNorm1d(out_node_nf)
        #self.graph_norm_in = nn.BatchNorm1d(in_node_nf)
        self.which_edges = [];

        for i in range(len(n_layers)):
            for j in range(n_layers[i]):
                self.which_edges.append(i);
                out_f = hidden_nf[i];
                if (i < len(n_layers) - 1) and j == n_layers[i]-1:
                    out_f = hidden_nf[i+1]
                self.graph_module.append(E_GCL(hidden_nf[i], out_f, hidden_nf[i], edges_in_d=in_edge_nf,
                                       act_fn=act_fn, residual=residual, attention=attention,
                                       normalize=normalize, tanh=tanh))

        self.drop1 = nn.Dropout(dropout_rate)

        self.n_fc_layers = num_FcLayers
        self.fc_post = nn.ModuleList();
        for i in range(num_FcLayers):
            self.fc_post.append( nn.Linear(out_node_nf, int(out_node_nf/2) ) );
        self.fc_times = num_Fc_times;
        self.fc_activation = torch.nn.LeakyReLU()

        self.final_fc = nn.Linear(num_FcLayers,output_dim)
        self.final_activation = nn.Sigmoid()

        self.out_node_nf = out_node_nf

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        #print(h.shape,edges[0][0].shape,x[0].shape)
        #h = self.graph_norm_in(h[0])
        h_prime = self.embedding_in0(h[0])
        x_prime = x[0]
        #print(edges[0][0].shape)

        for i in range(0, len(self.graph_module)):
            #print(i,h_prime.shape,h_prime,torch.any(torch.isnan(h_prime)))
            if torch.any(torch.isnan(h_prime)):
                exit()
            curr_edges = torch.LongTensor(edges[self.which_edges[i]])[0].to(self.device)
            curr_edge_feat = torch.FloatTensor(edge_attr[self.which_edges[i]])[0].to(self.device)
            h_prime, x_prime, _ = self.graph_module[i](h_prime, curr_edges, x_prime, edge_attr=curr_edge_feat)


        #print(h_prime)
        h_prime = self.embedding_out(h_prime)
        h_res = self.embedding_res(h[0]);
        h_prime = h_prime + h_res
        #print(h_prime.shape,h_prime)
        h_prime = self.graph_norm_end(h_prime)

        #print("h_prime:",h_prime)
        new_x = torch.empty((1,self.n_fc_layers),dtype=torch.float).to(self.device)
        #print(new_x)
        #print('outside',h_prime.shape,new_x.shape)
        #Now we do some funky dropout to get something to do stuff with
        for ii in range(self.n_fc_layers):
            #print("FC Layer: ",ii)
            curr_x = torch.empty((self.fc_times,1),dtype=torch.float).to(self.device)
            #print(curr_x)
            for jj in range(self.fc_times):
                curr_x[jj] = torch.mean( self.fc_post[ii]( self.drop1(h_prime) ) )
            #print(curr_x.shape)
            #print(torch.mean(curr_x))
            new_x[0,ii] = self.fc_activation( torch.mean(curr_x) )
            #new_x[0,ii] = torch.mean(curr_x)

        out = self.final_activation( self.final_fc(new_x) )

        return out

    def get_string_name(self):
        name = "IS_BINDER_hiddenNF-" + str(self.hidden_nf[0]);
        name += "_outNf-" + str(self.out_node_nf) + "x" + str(self.n_layers[0]);
        name += "_nFC-" + str(self.n_fc_layers);
        return name


class WHAT_CARB_BINDER(nn.Module):
    def __init__(self, in_node_nf=1280, hidden_nf=[196,196,196,196], out_node_nf=32, in_edge_nf=23,
            output_dim=17, dropout_rate=0.5, num_FcLayers=64, num_Fc_times=8,
            device='cpu', act_fn=nn.ReLU(), n_layers=[6,6,6,6],
            residual=True, attention=True, normalize=True, tanh=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features (array in this case of $4$)
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param output_dim: Number of output dimensions
        :param dropout_rate: Dropout rate post-graph
        :param num_drop_FcLayers: Number of FC layers post-graph
        :param num_Fc_times: Number of times each FC layers goes thru post-graph data
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(WHAT_CARB_BINDER, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding_in0 = nn.Linear(in_node_nf, self.hidden_nf[0])
        self.embedding_out = nn.Linear(self.hidden_nf[-1], out_node_nf)
        self.embedding_res = nn.Linear(in_node_nf, out_node_nf)
        self.graph_module = nn.ModuleList()

        self.graph_norm_end = nn.BatchNorm1d(out_node_nf)
        #self.graph_norm_in = nn.BatchNorm1d(in_node_nf)
        self.which_edges = [];

        self.n_layers = n_layers;

        for i in range(len(n_layers)):
            for j in range(n_layers[i]):
                self.which_edges.append(i);
                out_f = hidden_nf[i];
                if (i < len(n_layers) - 1) and j == n_layers[i]-1:
                    out_f = hidden_nf[i+1]
                self.graph_module.append(E_GCL(hidden_nf[i], out_f, hidden_nf[i], edges_in_d=in_edge_nf,
                                       act_fn=act_fn, residual=residual, attention=attention,
                                       normalize=normalize, tanh=tanh))

        self.drop1 = nn.Dropout(dropout_rate)

        self.n_fc_layers = num_FcLayers
        self.fc_post = nn.ModuleList();
        for i in range(num_FcLayers):
            self.fc_post.append( nn.Linear(out_node_nf, int(out_node_nf/2) ) );
        self.fc_times = num_Fc_times;
        self.fc_activation = torch.nn.LeakyReLU()

        self.final_fc = nn.Linear(num_FcLayers,output_dim)
        self.final_activation = nn.Sigmoid()

        self.out_node_nf=out_node_nf;

        self.to(self.device)

    def get_string_name(self):
        name = "WHAT_CARB_BINDER_hiddenNF-" + str(self.hidden_nf[0]);
        name += "_outNf-" + str(self.out_node_nf) + "x" + str(self.n_layers[0]);
        name += "_nFC-" + str(self.n_fc_layers);
        return name

    def forward(self, h, x, edges, edge_attr):
        #print(h.shape,edges[0][0].shape,x[0].shape)
        #h = self.graph_norm_in(h[0])
        h_prime = self.embedding_in0(h[0])
        x_prime = x[0]
        #print(edges[0][0].shape)

        for i in range(0, len(self.graph_module)):
            #print(i,h_prime.shape,h_prime,torch.any(torch.isnan(h_prime)))
            if torch.any(torch.isnan(h_prime)):
                print("h_prime is nan")
                exit()
            curr_edges = torch.LongTensor(edges[self.which_edges[i]])[0].to(self.device)
            curr_edge_feat = torch.FloatTensor(edge_attr[self.which_edges[i]])[0].to(self.device)
            h_prime, x_prime, _ = self.graph_module[i](h_prime, curr_edges, x_prime, edge_attr=curr_edge_feat)

        #print(h_prime)
        h_prime = self.embedding_out(h_prime)
        h_res = self.embedding_res(h[0]);
        h_prime = h_prime + h_res
        #print(h_prime.shape,h_prime)
        h_prime = self.graph_norm_end(h_prime)

        #print("h_prime:",h_prime)
        new_x = torch.empty((1,self.n_fc_layers),dtype=torch.float).to(self.device)
        #print(new_x)
        #print('outside',h_prime.shape,new_x.shape)
        #Now we do some funky dropout to get something to do stuff with
        for ii in range(self.n_fc_layers):
            #print("FC Layer: ",ii)
            curr_x = torch.empty((self.fc_times,1),dtype=torch.float).to(self.device)
            #print(curr_x)
            for jj in range(self.fc_times):
                curr_x[jj] = torch.mean( self.fc_post[ii]( self.drop1(h_prime) ) )
            #print(curr_x.shape)
            #print(torch.mean(curr_x))
            new_x[0,ii] = self.fc_activation( torch.mean(curr_x) )
            #new_x[0,ii] = torch.mean(curr_x)

        out = self.final_activation( self.final_fc(new_x) )

        return out


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":

    print("hello")
