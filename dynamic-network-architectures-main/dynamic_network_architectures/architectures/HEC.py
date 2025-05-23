import torch
import torch.nn as nn
from .unet import PlainConvUNet
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import nibabel as nib
import os





def dice_coefficient(pred, target, smooth=1e-5):
    if pred.is_cuda:
        # pred = np.copy(pred)
        pred = pred.cpu().numpy()
    else:
        pred = pred.numpy()

    if target.is_cuda:
        # target = np.copy(target)
        target = target.cpu().numpy()
    else:
        target = target.numpy()
    
   
    
    seg_slice1 = pred
    seg_slice2 = target
   

   
    pred_bin = (seg_slice1 > 0.0).astype(np.float32)
    target_bin = (seg_slice2 > 0.0).astype(np.float32)

    
    intersection = (pred_bin * target_bin).sum()
    dice = (2. * intersection + smooth) / (pred_bin.sum() + target_bin.sum() + smooth)

    return dice

class SlicePredictionMappingModule(nn.Module):
    def __init__(self, in_channels, similarity_threshold=0.6):
        super(SlicePredictionMappingModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  
        self.graph_conv = GraphConvolutionModule(in_channels)  
        self.similarity_threshold = nn.Parameter(torch.tensor([similarity_threshold], dtype=torch.float32))
 

    def forward(self, x):
        
        features = x
          
        batch_size, channels, depth, height, width = features.size()

        
        graph_features = features.view(batch_size * depth, channels, height, width)
        edge_index = self.create_dynamic_edge_index(graph_features, depth, height, width, channels)
        
        num_nodes = batch_size * depth
        
        
        graph_output = self.graph_conv(graph_features, edge_index, num_nodes)
        graph_output = graph_output.view(batch_size, channels, depth, height, width)
        
        return graph_output


    def create_dynamic_edge_index(self, features, depth, height, width, channels):
        num_nodes = features.size(0)
        device = features.device  

        
    
        edges_i = []
        edges_j = []
        
        for i in range(num_nodes-1):
                j = i + 1
                dice_score = dice_coefficient(features[i].squeeze(), features[j].squeeze(), smooth=1e-5)
                
                if dice_score > self.similarity_threshold.item() and dice_score < 1:
                    edges_i.append(i)
                    edges_j.append(j)
                    


        selected_i = torch.tensor(edges_i, device=device)
        selected_j = torch.tensor(edges_j, device=device)

        edges = torch.stack([torch.cat([selected_i, selected_j]), torch.cat([selected_j, selected_i])], dim=0)
        return edges.contiguous()


class GraphConvolutionModule(nn.Module):
    def __init__(self, channels, lower_threshold=0.0):
        super(GraphConvolutionModule, self).__init__()
        self.channels = channels
        self.lower_threshold = lower_threshold
        self.adaptive_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        

    def forward(self, x, edge_index, num_nodes):
        if edge_index.size(1) == 0:
            return x
            
            
            
      
        edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes=num_nodes, dtype=x.dtype, device=x.device)
        row, col = edge_index
        out = x.clone()

        
        for i in range(edge_index.size(1)):
            weighted_features = x[row[i]] * edge_weight[i]
            filtered_features = self.threshold_filter(weighted_features)


            
            adaptive_weighted_output = filtered_features * self.adaptive_weight
            out[col[i]] += adaptive_weighted_output


            
            # out[col[i]] += filtered_features  
            
            #out[col[i]] += x[row[i]] *  edge_weight[i]
                

        return out


    def gcn_norm(self, edge_index, num_nodes, dtype=None, device=None):
        
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=device)
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=dtype, device=device)
        deg[row] += edge_weight
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight
        
    def threshold_filter(self, slice_data):
        
        filtered_slice = torch.where((slice_data > self.lower_threshold) , slice_data, torch.zeros_like(slice_data))
        return filtered_slice





class MyUNet(PlainConvUNet):
    def __init__(self, input_channels, n_stages, features_per_stage, *args, **kwargs):
        super(MyUNet, self).__init__(input_channels, n_stages, features_per_stage, *args, **kwargs)
        self.slice_pred_map = SlicePredictionMappingModule(input_channels)  

    def forward(self, x):
        
        x = self.slice_pred_map(x)
        
        
        skips = self.encoder(x)
        x = self.decoder(skips)
        return x





