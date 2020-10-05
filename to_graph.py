import torch
import numpy as np
from torch_geometric.data import Data
import numpy as np
import scipy.spatial as ss

# construct a graph for tactile data
class TactileGraph(object):

    def __init__(self, k=0, useKNN=0, dist_threshold=0):
        # There are 39 taxles in each tactile finger, the coordinates are as follows:
        tact_coordinates = np.array([[-6, 0], [-5.3, -3], [-5.3, 3], [-4.6, -7.8], [-4.6, 7.8],
                    [-3.5, 0], [-3.05, -5.2], [-3.05, 5.2], [-3.1, -1.75], [-3.1, 1.75], # 10
                    [-1.6, -8.9], [-1.75, -3], [-1.6, 8.9], [-1.75, 3], [-1.5, 0],
                    [-0.7, -1.3], [-0.7, 1.3], [0, -6], [0, 6],[0, -3.5],
                    [0, 0], [0, 3.5], [0.8, -1.3], [0.8, 1.3], [1.6, -8.9],
                    [1.6, 8.9], [1.5, 0], [1.75, 3], [1.75, -3], [3.05, -5.2],
                    [3.1, 1.75], [3.05, 5.2], [3.1, -1.75], [3.6, 0], [4.6, -7.8], 
                    [4.6, 7.8], [5.3, -3], [5.3, 3], [6, 0]
                    ])
        assert(k >=0), 'For tactile graph, k should be non-negative'
        self.pos = tact_coordinates

        if k == 0: # use manual way to construct the graph
            self.edge_origins = np.array([1, 1, 1, 2, 3, 6, 2, 2, 7, 9, 3, 3, 8, 10, 4, 4, 7, 11,
                    5, 5, 8, 13, 6, 6, 6, 9, 10, 15, 7, 7, 12, 18, 8, 8, 14, 19, 9, 9, 12, 16, 
                    10, 10, 14, 15, 11, 11, 18, 25, 12, 12, 16, 20, 13, 13, 19, 26, 
                    14, 14, 17, 22, 15, 15, 15, 16, 17, 21, 16, 16, 21, 23, 17, 17, 21, 24, 
                    18, 18, 18, 20, 25, 30, 19, 19, 19, 22, 26, 32, 20, 20, 23, 29, 
                    21, 21, 21, 23, 24, 27, 22, 28, 23, 23, 27, 29, 24, 24, 24, 27, 28, 31, 
                    25, 35, 26, 36, 27, 27, 33, 34, 28, 28, 31, 32, 29, 29, 30, 33, 30, 30, 35, 37,
                    31, 31, 34, 38, 32, 32, 36, 38, 33, 33, 34, 37, 34, 39, 37, 39, 38, 39]) - 1 # since taxel number from 0 
            self.edge_ends    = np.array([2, 3, 6, 1, 1, 1, 7, 9, 2, 2, 8, 10, 3, 3, 7, 11, 4, 4, 
                    8, 13, 5, 5, 9, 10, 15, 6, 6, 6, 12, 18, 7, 7, 14, 19, 8, 8, 12, 16, 9, 9, 
                    14, 15, 10, 10, 18, 25, 11, 11, 16, 20, 12, 12, 19, 26, 13, 13, 
                    17, 22, 14, 14, 16, 17, 21, 15, 15, 15, 21, 23, 16, 16, 21, 24, 17, 17, 
                    20, 25, 30, 18, 18, 18, 22, 26, 32, 19, 19, 19, 23, 29, 20, 20,  
                    23, 24, 27, 21, 21, 21, 28, 22, 27, 29, 23, 23, 27, 28, 31, 24, 24, 24,
                    35, 25, 36, 26, 33, 34, 27, 27, 31, 32, 28, 28, 30, 33, 29, 29, 35, 37, 30, 30, 
                    34, 38, 31, 31, 36, 38, 32, 32, 34, 37, 33, 33, 39, 34, 39, 37, 39, 38]) - 1
        elif useKNN: 
            tree = ss.KDTree(tact_coordinates)
            _, idxs = tree.query(tact_coordinates, k = k+1) # including itself, so it is k+1
            idxs = idxs[:, 1:] # remove itself
            edge_origins = np.repeat(np.arange(len(tact_coordinates)), k)
            edge_ends = np.reshape(idxs, (-1))

            # make it undirected
            self.edge_origins = np.hstack((edge_origins, edge_ends))
            self.edge_ends = np.hstack((edge_ends, edge_origins))
       
        else: # use MST + sigma_d
            coordinates = self.pos # coordinates of taxels
            N = len(coordinates)
            self.edge_origins = []
            self.edge_ends = []
           
            visited_nodes = [20] # the center node number as 21
            unvisited_nodes = np.arange(0, N).tolist()
            unvisited_nodes.remove(20) # remove an item from the list
           
           # first do graph Kruskal's Minimum Spanning Tree algorithm
            while len(unvisited_nodes) > 0:
                min_dist = 100
                origin_index = -1
                end_index = -1
                for j in range(len(visited_nodes)):
                    dist = torch.norm(torch.from_numpy(coordinates[unvisited_nodes])-torch.from_numpy(coordinates[visited_nodes[j]]), dim=1, p=None)
                    [dist_min, index] = torch.sort(dist)      
                    if min_dist > dist_min[0]:
                        origin_index = visited_nodes[j]
                        min_dist = dist_min[0]
                        end_index = unvisited_nodes[index[0]]
                if end_index >= 0:
                    # add in pair
                    self.edge_origins.append(origin_index)
                    self.edge_ends.append(end_index)
                    
                    self.edge_origins.append(end_index)
                    self.edge_ends.append(origin_index)
                    visited_nodes.append(end_index)
                    unvisited_nodes.remove(end_index)

            A = torch.arange(0, N)
            C = torch.combinations(A, 2)
            for i in range(len(C)):
                c1 = torch.from_numpy(coordinates[C[i][0]])
                c2 = torch.from_numpy(coordinates[C[i][1]])
                dist = torch.norm(c1 - c2) #sqrt((c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]))
                if dist < dist_threshold:
                    self.edge_origins.append(C[i][0])
                    self.edge_ends.append(C[i][1])
                              
                    self.edge_origins.append(C[i][1])
                    self.edge_ends.append(C[i][0])
          


    def getEdge(self):
        edges = torch.tensor([self.edge_origins, self.edge_ends])
        return edges #self.edge_origins, self.edge_ends

    def __call__(self, sample):
        graph_x = sample
        graph_edge_index = torch.tensor([self.edge_origins, self.edge_ends], dtype=torch.long)
        graph_pos = self.pos
        data = Data(x=graph_x, edge_index = graph_edge_index, pos=graph_pos)
#        data = []
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
