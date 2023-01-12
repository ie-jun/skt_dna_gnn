import torch  
from torch import nn 
from torch.nn import functional as F
import numpy as np 

from source.layers.layers import *

# graph learning layer from connecting the dots.
class AdjConstructor(nn.Module): 
    r"""Constructs an adjacency matrix

    n_nodes: the number of nodes (node= cell)
    embedding_dim: dimension of the embedding vector
    """
    def __init__(self, n_nodes, embedding_dim, alpha= 3., top_k= 4): 
        super().__init__()
        self.emb1 = nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        self.emb2 = nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        self.theta1 = nn.Linear(embedding_dim, embedding_dim)
        self.theta2 = nn.Linear(embedding_dim, embedding_dim)
        self.alpha = alpha # controls saturation rate of tanh: activation function.
        self.top_k = top_k
    def forward(self, idx):
        emb1 = self.emb1(idx) 
        emb2 = self.emb2(idx) 

        emb1 = torch.tanh(self.alpha * self.theta1(emb1))
        emb2 = torch.tanh(self.alpha * self.theta2(emb2))

        adj_mat = torch.relu(torch.tanh(self.alpha*(emb1@emb2.T - emb2@emb1.T))) # adjacency matrix
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device) 
        mask.fill_(float('0'))
        if self.training:
            s1, t1 = (adj_mat + torch.rand_like(adj_mat)*0.01).topk(self.top_k, 1) # values, indices
        else: 
            s1, t1 = adj_mat.topk(self.top_k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj_mat = adj_mat * mask 
        return adj_mat

def encode_onehot(labels): 
    r""" Encode some relational masks specifying which vertices receive messages from which other ones.
    # Arguments          
    ___________             
    labels : np.array type 
    
    # Returns        
    _________          
    labels_one_hot : np.array type            
        adjacency matrix
    
    # Example-usage       
    _______________            
    >>> labels = [0,0,0,1,1,1,2,2,2]
    >>> labels_onehot = encode_onehot(labels)
    >>> labels_onehot 
    array(
        [[1, 0, 0],
         [1, 0, 0],             
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 1],            
         [0, 0, 1]], dtype=int32)      
    """
    classes = set(labels) 
    classes_dict = {c: np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype= np.int32)
    return labels_onehot

def generate_fcn(num_objects, device= None): 
    r"""Generates fcn (Fully Connected Graph)
    
    # Arguments          
    ___________                
    num_objects : int
        the number of objects               
    device : torch.device     
        device - cpu or cuda    

    # Returns    
    _________     
    rel_rec : torch.FloatTensor    
        relation-receiver     
    rel_send : torch.FloatTensor    
        relation-receiver     
    """
    fcn = np.ones([num_objects, num_objects])
    
    rec, send = np.where(fcn)
    rel_rec = np.array(encode_onehot(rec), dtype= np.float32)
    rel_send = np.array(encode_onehot(send), dtype= np.float32)
    
    rel_rec = torch.FloatTensor(rel_rec) if device is None \
        else torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send) if device is None \
        else torch.FloatTensor(rel_send).to(device)
    
    return rel_rec, rel_send  

# Graph Learning Layer - Encoder 
class GraphLearningEncoder(nn.Module): 
    r""" Encoder module using TemporalConvolutionModule defined in layers.py 
    # Arguments
    ___________
    num_heteros : int 
        the number of heterogeneous groups      
    
    # forwards
    __________
    returns adjacency matrix for every item in a batch

    """
    def __init__(self, num_heteros, time_lags, num_time_series, **kwargs): 
        super().__init__() 
        self.num_heteros = num_heteros
        self.time_lags = time_lags
        self.tcm = nn.Sequential(
            TemporalConvolutionModule(num_heteros, num_heteros, num_heteros=num_heteros, num_time_series= num_time_series),
            nn.Conv2d(num_heteros, num_heteros, (time_lags, 1), groups= num_heteros)) 
        self.node2edge_conv = nn.Conv2d(num_heteros, num_heteros, (1, 2), groups= num_heteros)
        self.edge2node_conv = nn.Conv2d(num_heteros, num_heteros, (1, 1), groups= num_heteros)
        self.node2edge_conv_2 = nn.Conv2d(num_heteros, num_heteros, (1, 3), groups= num_heteros) 
        self.conv_out = nn.Conv2d(num_heteros, num_heteros, (1, time_lags))

    def edge2node(self, x, rel_rec, rel_send):
        # fully-connected-graph. 
        incoming = torch.matmul(rel_rec.permute(0,2,1), x) # (c, n, n^2) x (bs, c, n^2, 1) --> (bs, c, n, 1)
        return incoming / incoming.size(2)

    def node2edge(self, x, rel_rec, rel_send):
        # fully-connected-graph
        receivers = torch.matmul(rel_rec, x) # (c, n^2, n) x (bs, c, n, 1) --> (bs, c, n^2, 1)
        senders = torch.matmul(rel_send, x) # (c, n^2, n) x (bs, c, n, 1) --> (bs, c, n^2, 1)
        edges = torch.cat([senders, receivers], dim=-1) # (bs, c, n^2, 2)
        return edges
    
    def forward(self, x, rel_rec, rel_send): 
        r"""
        # forwards
        __________
        feed-forwards works as follows...     
        x : torch.FloatTensor     
            shape of x is 'bs x c x t x n'

        (1) TemporalConvolutionModule   
        h :nn.Conv2d(tcm(x))  
            shape of h is 'bs x c x 1 x n'
            reshape so that, 
            the shape of h is 'bs x c x n x 1' 
            'n' is the number of 'nodes' 
        
        (2) Node2Edge operation 
        h_e = [rel_rec @ h; rel_send @ h]
            the shape of h_e is 'bs x c x n^2 x 2
        h_e = conv2d(h_e)
            the shape of h_e is 'bs x c x n^2 x 1
        h_e_skip = h_e 

        (3) Edge2Node operation 
        h_n = rel_rec.t @ h_e 
            the shape of h_n is 'bs x c x n x 1'
        h_n = conv2d(h_n) 
            the shape of h_n is 'bs x c x n x 1' 
        
        (4) Node2Edge operation 
        h_e = [rel_rec @ h_n ; rel_send @ h_n] 
            the shape of h_e is 'bs x c x n^2 x 2'            
                     
        (5) Skip connection 
        h_e = [h_e; h_e_skip] 
            the shape of h_e is 'bs x c x n^2 x 3'    
        h_e = conv2d(h_e)     
            the shape of h_e is 'bs x c x n^2 x 1'    

        (6) reshape the logits
        adj = h_e.reshape(bs, c, n, n) 
        """
        bs, c, t, n = x.shape
        # (1)
        h = self.tcm(x).permute(0, 1, 3, 2) # bs, c, n, 1 
        # print(f'(1) {h.shape}')
        # (2) 
        h = self.node2edge(h, rel_rec, rel_send) # bs, c, n^2, 2
        h = self.node2edge_conv(h) # bs, c, n^2, 1 
        h_skip = h # bs, c, n^2, 1 
        # print(f'(2) {h.shape}')
        # (3) 
        h = self.edge2node(h, rel_rec, rel_send) # bs, c, n, 1 
        h = self.edge2node_conv(h) # bs, c, n, 1 
        # print(f'(3) {h.shape}')
        # (4) 
        h = self.node2edge(h, rel_rec, rel_send) # bs x c x n^2 x 2
        # print(f'(4) {h.shape}')
        # (5) 
        h = torch.concat((h, h_skip), dim= -1) # bs x c x n^2 x 3 
        h = self.node2edge_conv_2(h) # bs x c x n^2 x 1
        # print(f'(5) {h.shape}')
        h = h.squeeze().reshape((bs, c, n, n)) # bs x c x n x n
        return h

#graph learning layer using recurrent property
class RecurrentGraphLearningEncoder(GraphLearningEncoder):
    r""" Change GraphLearningEncoder to be recurrent.
    # Arguments
    ___________
    num_heteros : int
        the number of heterogeneous groups

    # forwards
    __________
    returns adjacency matrix for every item in a batch

    """

    def __init__(self, num_heteros, time_lags, num_time_series, **kwargs):
        super().__init__(num_heteros, time_lags, num_time_series, **kwargs)
        '''
        num_heteros,time_lags,tcm,node2edge_conv,edge2node_conv,node2edge_conv_2,conv_out,edge2node,node2edge
        are already initialized in GraphLearningEncoder.
        '''
        self.gru =nn.GRU(input_size =num_heteros, hidden_size=num_heteros*2)
        self.gru_fc =nn.Linear(num_heteros*2,num_heteros)

    def forward(self, x, rel_rec, rel_send):
        r"""
        # forwards
        __________
        feed-forwards works as follows...
        x : torch.FloatTensor
            shape of x is 'bs x c x t x n'

        (1) TemporalConvolutionModule
        h :nn.Conv2d(tcm(x))
            shape of h is 'bs x c x 1 x n'
            reshape so that,
            the shape of h is 'bs x c x n x 1'
            'n' is the number of 'nodes'

        (2) Node2Edge operation
        h_e = [rel_rec @ h; rel_send @ h]
            the shape of h_e is 'bs x c x n^2 x 2
        h_e = conv2d(h_e)
            the shape of h_e is 'bs x c x n^2 x 1
        h_e_skip = h_e

        (3) Edge2Node operation
        h_n = rel_rec.t @ h_e
            the shape of h_n is 'bs x c x n x 1'

        (4)Recurrent part is added        (What's different with GraphLearningEncoder)
        fc_input = GRU(h_n)
            the shape of h_e is 'bs x c x 2n x 1   (detail process is in forward)
        out = Linear(fc_input)
            the shape of h_e is 'bs x c x n x 1   (detail process is in forward)


        (5)Edge2Node_conv is applied
        h_n = conv2d(out)
            the shape of h_n is 'bs x c x n x 1'



        (5) Node2Edge operation
        h_e = [rel_rec @ h_n ; rel_send @ h_n]
            the shape of h_e is 'bs x c x n^2 x 2'

        (6) Skip connection
        h_e = [h_e; h_e_skip]
            the shape of h_e is 'bs x c x n^2 x 3'
        h_e = conv2d(h_e)
            the shape of h_e is 'bs x c x n^2 x 1'

        (7) reshape the logits
        adj = h_e.reshape(bs, c, n, n)
        """
        bs, c, t, n = x.shape
        # (1)
        h = self.tcm(x).permute(0, 1, 3, 2)  # bs, c, n, 1
        # print(f'(1) {h.shape}')
        # (2)
        h = self.node2edge(h, rel_rec, rel_send)  # bs, c, n^2, 2
        h = self.node2edge_conv(h)  # bs, c, n^2, 1
        h_skip = h  # bs, c, n^2, 1
        # (3)
        h = self.edge2node(h, rel_rec, rel_send)  # bs, c, n, 1
        # (4)
        h = h.reshape(-1,h.size(2),h.size(3)) # bs*c, n, 1
        recurrent_input = h.permute(0,2,1) # bs*c, 1, n
        out, _ = self.gru(recurrent_input)
        fc_input = out[:,-1,:].unsqueeze(1) # bs*c , 1, 2n
        fc_out = self.gru_fc(fc_input) # bs*c , 1, n
        fc_out = fc_out.permute(0,2,1) # bs*c , n, 1
        fc_out = fc_out.reshape(bs,c,n,1) # bs, c, n, 1
        h = fc_out
        # (5)
        h = self.edge2node_conv(h)  # bs, c, n, 1
        # (6)
        h = self.node2edge(h, rel_rec, rel_send)  # bs x c x n^2 x 2
        # (7)
        h = torch.concat((h, h_skip), dim=-1)  # bs x c x n^2 x 3
        h = self.node2edge_conv_2(h)  # bs x c x n^2 x 1
        h = h.squeeze().reshape((bs, c, n, n))  # bs x c x n x n
        return h





class GraphLearningEncoderModule(nn.Module): 
    r"""GraphLearningEncoderModule    
    VAE is used as a graph learning encoder.   
    It uses 2D-group-convolution to send and aggregate messages from nodes and edges     
    # Arguments       
    ___________              
    num_heteros : int    
        the number of heterogeneous groups (stack along the channel dimension)
    time_lags: int 
        the size of 'time_lags'       
    num_ts : int     
        the number of time-series    
        should be 10 for the skt-data     
    """

    def __init__(self, num_heteros, time_lags, num_ts, device,graph_type,**kwargs):
        super().__init__()

        # generates fully-connected-graph
        rel_rec, rel_send = [], []
        for i in range(num_heteros): 
            rec, send = generate_fcn(num_ts)
            rel_rec.append(rec); rel_send.append(send) 
        self.rel_rec = torch.stack(rel_rec, dim= 0).to(device)
        self.rel_send = torch.stack(rel_send, dim= 0).to(device)
        
        if graph_type == 'heteroNRI':
            self.gle = GraphLearningEncoder(num_heteros, time_lags, num_ts, **kwargs)
        elif graph_type == 'heteroNRI_gru':
            self.gle = RecurrentGraphLearningEncoder(num_heteros, time_lags, num_ts, **kwargs)

        self.num_heteros, self.time_lags, self.num_ts = num_heteros, time_lags, num_ts
        # self.tau = tau
        # self.hard = hard

    def forward(self, x): 
        logits = self.gle(x, self.rel_rec, self.rel_send)
        return logits

