from abc import ABC, abstractmethod
from typing import Tuple
import os
import torch
from torch import nn
import numpy as np

class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx

                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size] # 500, 4
                        q = self.get_queries(these_queries) # 500, 400
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        scores = q @ rhs 
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

class TCompoundQ(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,no_time_emb=False, init_size: float = 1e-2):
        super(TCompoundQ, self).__init__()
        print(os.path.basename(__file__))
        self.sizes = sizes
        self.rank = rank
        self.W = nn.Embedding(2*rank, 1, sparse=True)
        self.W.weight.data *= 0

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * rank, sparse=True),
            nn.Embedding(sizes[1], 5 * rank, sparse=True),
            nn.Embedding(sizes[3], 4 * rank, sparse=True)
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

    @staticmethod
    def has_time():
        return True

    def _calc(self, h, r):
        w_a, x_a, y_a, z_a = torch.chunk(h, 4, dim=-1)
        w_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=-1)

        A = complex_mul(w_a, w_b) - complex_mul(x_a, x_b) - complex_mul(y_a, y_b) - complex_mul(z_a, z_b)
        B = complex_mul(w_a, x_b) + complex_mul(x_a, w_b) + complex_mul(y_a, z_b) - complex_mul(z_a, y_b)
        C = complex_mul(w_a, y_b) - complex_mul(x_a, z_b) + complex_mul(y_a, w_b) + complex_mul(z_a, x_b)
        D = complex_mul(w_a, z_b) + complex_mul(x_a, y_b) - complex_mul(y_a, x_b) + complex_mul(z_a, w_b)

        return torch.cat([A, B, C, D], dim=-1)

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        rel, theta = rel[:, :4 * self.rank], rel[:, 4 * self.rank:]
        rel = rel[:, :self.rank * 2], rel[:, self.rank * 2:]
        time = time[:, :self.rank * 2], time[:, self.rank * 2:]

        rel = rel[0] + time[0], rel[1]
        full_rel = self._calc(rel[0], time[1]), rel[1]

        theta = theta / (1 / self.pi)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        lhs = lhs.view(lhs.shape[0], -1, 2)

        lhs_rotated = torch.cat((cos * lhs[:, :, 0:1], sin * lhs[:, :, 0:1]), dim=-1)
        lhs_rotated += torch.cat((-sin * lhs[:, :, 1:], cos * lhs[:, :, 1:]), dim=-1)
        lhs_rotated = lhs_rotated.view(lhs.shape[0], -1)

        lhs_rotated += full_rel[1]
        res = self._calc(lhs_rotated, full_rel[0])
        return torch.sum(res * rhs, dim=1, keepdim=True)

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        rel, theta = rel[:, :4 * self.rank], rel[:, 4 * self.rank:]
        rel = rel[:, :self.rank * 2], rel[:, self.rank * 2:]
        time = time[:, :self.rank * 2], time[:, self.rank * 2:]

        rel = rel[0] + time[0], rel[1]
        full_rel = self._calc(rel[0], time[1]), rel[1]

        right = self.embeddings[0].weight

        theta = theta / (1 / self.pi)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        lhs = lhs.view(lhs.shape[0], -1, 2)

        lhs_rotated = torch.cat((cos * lhs[:, :, 0:1], sin * lhs[:, :, 0:1]), dim=-1)
        lhs_rotated += torch.cat((-sin * lhs[:, :, 1:], cos * lhs[:, :, 1:]), dim=-1)
        lhs_rotated = lhs_rotated.view(lhs.shape[0], -1)

        lhs_rotated += full_rel[1]
        res = self._calc(lhs_rotated, full_rel[0])

        regularizer = [(get_norm(lhs_rotated, 8), get_norm(full_rel[0], 8), get_norm(rhs, 8))]

        return (
                res @ right.t()
               ), regularizer,  self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight


    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1]) 
        time = self.embeddings[2](queries[:, 3])

        rel, theta = rel[:, :4 * self.rank], rel[:, 4 * self.rank:]
        rel = rel[:, :self.rank * 2], rel[:, self.rank * 2:]
        time = time[:, :self.rank * 2], time[:, self.rank * 2:]

        rel = rel[0] + time[0], rel[1]
        full_rel = self._calc(rel[0], time[1]), rel[1]

        theta = theta / (1 / self.pi)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        lhs = lhs.view(lhs.shape[0], -1, 2)

        lhs_rotated = torch.cat((cos * lhs[:, :, 0:1], sin * lhs[:, :, 0:1]), dim=-1)
        lhs_rotated += torch.cat((-sin * lhs[:, :, 1:], cos * lhs[:, :, 1:]), dim=-1)
        lhs_rotated = lhs_rotated.view(lhs.shape[0], -1)

        lhs_rotated += full_rel[1]
        res = self._calc(lhs_rotated, full_rel[0])
        return res

def complex_mul(a, b):
    assert a.size(-1) == b.size(-1)
    dim = a.size(-1) // 2
    a_1, a_2 = torch.split(a, dim, dim=-1)
    b_1, b_2 = torch.split(b, dim, dim=-1)

    A = a_1 * b_1 - a_2 * b_2
    B = a_1 * b_2 + a_2 * b_1

    return torch.cat([A,B], dim=-1)

def get_norm(x, nums):
    dim = x.size(-1) // nums
    res = torch.split(x, dim, dim=-1)
    norm = 0
    for i in res:
        norm += i**2
    return torch.sqrt(norm)