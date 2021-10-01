import numpy as np
import cv2
import tools.utils.openpose as ops
import json
import os
import torch

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    sample_name = []
    lines=[]

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)
    def __str__(self):
        return self.A
    def setSim(self,sample_name):
        self.sample_name=sample_name
    def getSim(self):
        return self.sample_name

    def match_train(self, file):
        files = "D:/ljm/1/st-gcn-master/data/data/kinetics"
        json_file = os.listdir(files)
        for i in range(len(json_file)):
            if (file == json_file[i]):
                full_name = os.path.join(files, json_file[i])
                return full_name
    def getSlectIndex(self,file):
         fr = open(file,'r',encoding='utf-8')
         pi=[]
         temp = json.load(fr)
         data = temp['data']
         for bean in data:
             pi.append(bean['frame_index'])
         return pi;

    def getpoints(self):
      peis = []
      for i in range(len(self.lines)):
         if(i%2)==0:
            peis.append((int(self.lines[i]),int(self.lines[i+1])))
      return peis
    delaunay_color = (255, 255, 255)
    def getframe_index(self,file,frame_index):
        fr = open(file, 'r', encoding='utf-8')
        temp = json.load(fr)
        data = temp['data']
        peis=[]
        for frame in data:
            if (frame['frame_index'] == frame_index):
                frameinfo = frame['skeleton']
                for pose in frameinfo:
                    points = pose['pose']
                    for i in range(len(points)):
                        if (i % 2) == 0:
                            peis.append((int(points[i] * 340), int(points[i + 1] * 256)))
        return peis



    def calculateDelaunayTriangles(self,rect, peis):
        subdiv = cv2.Subdiv2D(rect);
        for p in peis:
            subdiv.insert(p)
        triangleList = subdiv.getTriangleList();
        delaunayTri = []
        pt = []
        count= 0
        for t in triangleList:
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
                count = count + 1
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(peis)):
                        if(abs(pt[j][0] - peis[k][0]) < 1.0 and abs(pt[j][1] - peis[k][1]) < 1.0):
                            ind.append(k)
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1]))
                    delaunayTri.append((ind[0], ind[2]))
                    delaunayTri.append((ind[1], ind[2]))
            pt = []
        return delaunayTri
    def getFile(self,files):
        path=[]
        for root,dir,files in os.walk(files):
            for f in files:
                path.append(os.path.join(root,f))
        return path
    def get_edge(self, layout):
        if layout == 'openpose':
            rect = (0, 0, 340, 256)
            subdiv = cv2.Subdiv2D(rect)
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            '''neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1'''
            files = "D:/ljm/1/st-gcn-master/data/data/kinetics"
            for filename in os.listdir(files):
              full_name = os.path.join(files,filename)
              pis = self.getSlectIndex(full_name)
              pis.sort()
              for frame in pis:
                 peis= self.getframe_index(full_name,frame)
            rect = (0, 0, 340, 256)
            neighbor_link = self.calculateDelaunayTriangles(rect, peis)
            self.edge = neighbor_link + self_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD