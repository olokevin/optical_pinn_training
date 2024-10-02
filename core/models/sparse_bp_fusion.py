from typing_extensions import final
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
# from tensor_layers.layers import TensorizedLinear,TTM_Linear_module
from .sparse_bp_ttm_mlp import TTM_Linear_module
from .sparse_bp_attn import Encoder
# from tensor_layers.Transformer_tensor import Encoder, TextSubNet_attention

__all__ = ["SparseBP_MZI_LMF"]

class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''


    def __init__(self, in_size, hidden_size, out_size, dropout, shape, max_rank=20, tensor_type = 'TTM', device=None, dtype=None):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(SubNet,self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.prior_type='log_uniform'
        self.eta=1.0

        self.shape_1 = shape[0]
        self.shape_2 = shape[1]
        self.shape_3 = shape[2]

        self.USE_TT_LINEAR = 1
        if self.USE_TT_LINEAR == 1:
          # self.linear_1 = AdaptiveRankLinear(in_size, hidden_size, max_rank = max_rank,  min_dim = 2, 
          #                                       bias=True, tensor_type=tensor_type, prior_type='log_uniform',
          #                                       eta=None, device=device, dtype=dtype)
          # self.linear_2 = AdaptiveRankLinear(hidden_size, hidden_size, max_rank=max_rank, min_dim = 2, 
          #                                     bias = True, tensor_type=tensor_type,
          #                                     prior_type='log_uniform', 
          #                                     eta=None, device=device, dtype=dtype)
          # self.linear_3 = AdaptiveRankLinear(hidden_size, out_size, max_rank=max_rank, min_dim = 2, 
          #                                     bias = True, tensor_type=tensor_type,
          #                                     prior_type='log_uniform', 
          #                                     eta=None, device=device, dtype=dtype)
          self.linear_1 = TTM_Linear_module(in_size, hidden_size, bias=None, shape=self.shape_1, tensor_type=tensor_type, max_rank=max_rank,
                          prior_type=self.prior_type, eta=self.eta, device=device, dtype=dtype)
          self.linear_2 = TTM_Linear_module(hidden_size, hidden_size, bias=None, shape=self.shape_2, tensor_type=tensor_type, max_rank=max_rank,
                          prior_type=self.prior_type, eta=self.eta, device=device, dtype=dtype)
          self.linear_3 = TTM_Linear_module(hidden_size, out_size, bias=None, shape=self.shape_3, tensor_type=tensor_type, max_rank=max_rank,
                          prior_type=self.prior_type, eta=self.eta, device=device, dtype=dtype)
        else:  
          self.linear_1 = nn.Linear(in_size, hidden_size)
          self.linear_2 = nn.Linear(hidden_size, hidden_size)
          self.linear_3 = nn.Linear(hidden_size, out_size)

    def forward(self,x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped= self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

# class TextSubNet_LSTM(nn.Module):
#     '''
#     The LSTM-based subnetwork that is used in LMF for text
#     '''

#     def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False, max_rank=20, tensor_type = 'TTM', device=None, dtype=None):
#         '''
#         Args:
#             in_size: input dimension
#             hidden_size: hidden layer dimension
#             num_layers: specify the number of layers of LSTMs.
#             dropout: dropout probability
#             bidirectional: specify usage of bidirectional LSTM
#         Output:
#             (return value in forward) a tensor of shape (batch_size, out_size)
#         '''
#         super(TextSubNet_LSTM,self).__init__()
#         self.USE_TT_LINEAR = 1
#         # self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
#         self.rnn = AdaptiveRankLSTM(in_size, hidden_size, bias = True,
#                                               max_rank=max_rank, tensor_type=tensor_type,
#                                               prior_type='log_uniform', eta=None,
#                                               device=device, dtype=dtype)
#         self.dropout = nn.Dropout(p=dropout)

#         if self.USE_TT_LINEAR == 1:
#           self.linear_1 = AdaptiveRankLinear(hidden_size, out_size, bias = True, 
#                                               min_dim = 2,
#                                               max_rank=max_rank, 
#                                               tensor_type=tensor_type,
#                                               prior_type='log_uniform', 
#                                               eta=None, device=device, dtype=dtype)
#         else:
#           self.linear_1 = nn.Linear(hidden_size,out_size)

#     def forward(self,x):
#         '''
#         Args:
#             x: tensor of shape (batch_size, sequence_len, in_size)
#         '''
#         _, final_states = self.rnn(x)
#         h = self.dropout(final_states[0].squeeze())
#         y_1 = self.linear_1(h)
#         return y_1

class TextSubNet_attention(nn.Module):
    ''' The attention-based subnetwork that is used in LMF for text '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrainMatrix',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrainMatrix',
            d_classifier=512,num_class = 2, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrainMatrix',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized)
        
        # self.preclassifier = nn.Linear(d_model,d_classifier)

        self.classifier = nn.Linear(d_classifier,num_class)

        if tensorized==True:
            self.preclassifier = TTM_Linear_module(d_model, d_classifier, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
        else:
            self.preclassifier = nn.Linear(d_model,d_classifier)

        self.dropout = nn.Dropout(p=dropout_classifier)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, src_seq, src_mask=None, return_attns=False):

        x = self.encoder(src_seq, src_mask, seg=None)
        # print(x)
        # x = torch.squeeze(x[:,0,:])
        x = torch.mean(x,1)
        # x = self.dropout(x)
        # x = self.preclassifier(x)
        x = self.classifier(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.dropout(x)
        # x = self.classifier(x)
        return x

class FusionLayer(nn.Module):
    # def __init__(self, input_dims, hidden_dims, audio_out, video_out, text_out, dropouts, output_dim, rank):
    def __init__(self, input_dims, hidden_dims, sub_out_dims, dropouts, shapes, output_dim, rank, max_rank=2, TT_FUSION = 1, tensor_type = 'TTM', device=None, dtype=None):
        
        super(FusionLayer,self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]    

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]

        self.audio_out= sub_out_dims[0]
        self.video_out= sub_out_dims[1]
        self.text_out = sub_out_dims[2]

        self.audio_shape= shapes[0]
        self.video_shape= shapes[1]
        self.text_shape = shapes[2]

        self.output_dim = output_dim
        self.rank = rank

        self.post_fusion_prob = dropouts[3]
        self.prior_type='log_uniform'
        self.eta=1.0

        self.TT_FUSION = TT_FUSION
        
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

        if(self.TT_FUSION == 0):
          self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_out, self.output_dim))
          self.video_factor = Parameter(torch.Tensor(self.rank, self.video_out, self.output_dim))
          self.text_factor  = Parameter(torch.Tensor(self.rank, self.text_out, self.output_dim))
          # init factors
          xavier_normal_(self.audio_factor)
          xavier_normal_(self.video_factor)
          xavier_normal_(self.text_factor)
        else:
          # self.audio_fusion = AdaptiveRankLinear(in_features = self.audio_out, out_features = rank*output_dim, max_rank = max_rank, 
          #                                         min_dim=2, bias=False, tensor_type=tensor_type, prior_type='log_uniform',
          #                                         eta=None, device=device, dtype=dtype)
          # self.video_fusion = AdaptiveRankLinear(in_features = self.video_out, out_features = rank*output_dim, max_rank = max_rank, 
          #                                         min_dim=2, bias=False, tensor_type=tensor_type, prior_type='log_uniform',
          #                                         eta=None, device=device, dtype=dtype)
          # self.text_fusion  = AdaptiveRankLinear(in_features = self.text_out, out_features = rank*output_dim, max_rank = max_rank, 
          #                                         min_dim=2, bias=False, tensor_type=tensor_type, prior_type='log_uniform',
          #                                         eta=None, device=device, dtype=dtype)
          
          self.audio_fusion = TTM_Linear_module(self.audio_out, rank*output_dim, bias=None, shape=self.audio_shape, 
                              tensor_type=tensor_type, max_rank=max_rank, prior_type=self.prior_type, eta=self.eta, device=device, dtype=dtype)
          self.video_fusion = TTM_Linear_module(self.video_out, rank*output_dim, bias=None, shape=self.video_shape, 
                              tensor_type=tensor_type, max_rank=max_rank, prior_type=self.prior_type, eta=self.eta, device=device, dtype=dtype)
          self.text_fusion  = TTM_Linear_module(self.text_out,  rank*output_dim, bias=None, shape=self.text_shape, 
                              tensor_type=tensor_type, max_rank=max_rank, prior_type=self.prior_type, eta=self.eta, device=device, dtype=dtype)

        
    
    def forward(self, audio_h,video_h,text_h):
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        #concat 1 to unimodal representations 
        # dim 0: batch dim 1: vector, add 1s in vector (dim 1)
        _audio_h = torch.cat(audio_h[:-1],(Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False)), dim=1)
        _video_h = torch.cat(video_h[:-1],(Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False)), dim=1)
        _text_h  = torch.cat(text_h[:-1], (Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False)), dim=1)

        # forward

        # non tensorized
        # (batch, audio_out) * (rank, audio_out, output_dim) = (rank, batch, output_dim)

        if(self.TT_FUSION == 0):
          fusion_audio = torch.matmul(_audio_h, self.audio_factor)
          fusion_video = torch.matmul(_video_h, self.video_factor)
          fusion_text = torch.matmul(_text_h, self.text_factor)
          fusion_zy = fusion_audio * fusion_video * fusion_text
          # summation over each rank output (rank in dim 0, now we eliminate rank, thus sum over dim 0)
          output = torch.sum(fusion_zy, dim=0).squeeze()

        else: 
          fusion_audio = torch.reshape(self.audio_fusion(_audio_h), (batch_size, self.rank, self.output_dim))
          fusion_video = torch.reshape(self.video_fusion(_video_h), (batch_size, self.rank, self.output_dim))
          fusion_text  = torch.reshape(self.text_fusion(_text_h),   (batch_size, self.rank, self.output_dim))

          # element-wise product over all output_dim
          fusion_zy = fusion_audio * fusion_video * fusion_text

          # print("fusion_zy size",fusion_zy.size())

          # summation over each rank output (rank in dim 1, now we eliminate rank, thus sum over dim 1)
          output = torch.sum(fusion_zy, dim=1).squeeze()

          # print("output size",output.size())

        # use linear transformation instead of simple summation, more flexibility
        #output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        #output = output.view(-1, self.output_dim)

        return output

class SparseBP_MZI_LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, sub_out_dims, n_layers, n_head, dropouts, output_dim, rank, max_rank, TT_FUSION, use_LSTM, tensor_type, device, dtype, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            audio_out - int, specifying the resulting dimensions of the audio subnetwork
            video_out - int, specifying the resulting dimensions of the video subnetwork
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF (for fusion layer)
            max_rank - max_rank for subnetworks
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''

        super(SparseBP_MZI_LMF,self).__init__()

        self.device = device

		# dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2] 
        self.text_seq = input_dims[3]     

        self.n_layers = n_layers
        self.n_head   = n_head
        self.d_q = self.text_in//self.n_head
        self.d_k = self.text_in//self.n_head
        self.d_v = self.text_in//self.n_head
        self.d_model = self.text_in
        self.d_inner = self.text_in
        # self.d_inner = 4 * self.text_in
        self.pad_idx = None

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]

        self.audio_out= sub_out_dims[0]
        self.video_out= sub_out_dims[1]
        self.text_out = sub_out_dims[2]

        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.TT_FUSION = TT_FUSION

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        self.use_LSTM = use_LSTM

        #define pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_out, self.audio_prob, 
                            shape=[[[2,4,2,5],[4,2,2,2]], [[2,2,2,4],[4,2,2,2]], [[2,2,2,4],[4,2,2,2]]],  
                            max_rank=max_rank, tensor_type=tensor_type, device=device, dtype=dtype)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_out, self.video_prob,
                            shape=[[[3,2,2,3],[4,2,2,2]], [[2,2,2,4],[4,2,2,2]], [[2,2,2,4],[4,2,2,2]]],  
                            max_rank=max_rank, tensor_type=tensor_type, device=device, dtype=dtype)
        
        if self.use_LSTM == 1:
           raise NotImplementedError
          # self.text_subnet = TextSubNet_LSTM(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob, max_rank=max_rank, tensor_type=tensor_type, device=device, dtype=dtype)
        else:
          self.text_subnet = TextSubNet_attention(
              n_src_vocab=self.text_seq, d_word_vec=self.text_in, n_layers=self.n_layers, n_head=self.n_head, d_q=self.d_q, d_k=self.d_k, d_v=self.d_v,
              d_model=self.d_model, d_inner=self.d_inner, pad_idx=None, dropout=self.text_prob, n_position=self.text_seq, scale_emb=False,
              emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
              attention_shape = [[[5,4,3,5],[5,4,3,5]],[[5,4,3,5],[5,4,3,5]]], attention_rank = [max_rank,max_rank],attention_tensor_type = 'TensorTrainMatrix',
              ffn_shape = [[[5,4,3,5],[5,4,3,5]],[[5,4,3,5],[5,4,3,5]]], ffn_rank = [max_rank,max_rank],ffn_tensor_type = 'TensorTrainMatrix',
              d_classifier=self.d_inner, num_class = self.text_out, dropout_classifier = 0.2,
              classifier_shape = [[5,4,3,5],[5,4,3,5]],classifier_rank = max_rank,classifier_tensor_type = 'TensorTrainMatrix',
              bit_attn = 8, scale_attn = 2**(-5), 
              bit_ffn = 8, scale_ffn = 2**(-5),
              bit_a = 8, scale_a = 2**(-5),
              quantized = False,
              tensorized=True)

        self.fusion = FusionLayer(input_dims, hidden_dims, sub_out_dims, dropouts,
                      shapes=[[[4,2,2,2],[rank,1,output_dim,1]],[[4,2,2,2],[rank,1,output_dim,1]],[[4,2,2,2],[rank,1,output_dim,1]]],
                      output_dim=output_dim, rank=rank, max_rank=max_rank, TT_FUSION=TT_FUSION, tensor_type=tensor_type, device=device, dtype=dtype)
       

    def forward(self, x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_x = x[0]
        video_x = x[1]
        text_x  = x[2]

        audio_h  = self.audio_subnet(audio_x)
        video_h  = self.video_subnet(video_x)
        if self.use_LSTM == 1:
          text_h   = self.text_subnet(text_x)
        else:
          # remove additional src_mask building parts in encoder forward
          src_mask = None
          text_h   = self.text_subnet(text_x, src_mask, return_attns=False)
        
        output = self.fusion(audio_h,video_h,text_h)

        if self.use_softmax:
            output = F.softmax(output)
        return output



 
