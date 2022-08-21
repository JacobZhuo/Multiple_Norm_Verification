import numpy as np
import torch
from torch import nn



class BoundCalculator:

    def __init__(self, model, in_shape, in_min, in_max):
        super(BoundCalculator, self).__init__()

        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()
        in_numel = in_shape
        shapes.append(in_numel)
        for name,param in model.named_parameters():
            if 'weight' in name:
                Ws.append(param.data.numpy())
            if 'bias' in name:
                bs.append(param.data.numpy())
                shapes.append(param.numel())

        self.in_min, self.in_max = in_min, in_max

        self.in_numel = in_numel
        self.num_layers = num_layers
        self.shapes = shapes
        self.Ws = Ws
        self.bs = bs

        self.l = None
        self.u = None

    def verify(self, y_true, y_adv):
        """
            Assert if y_true >= y_adv holds for all
        :param y_true:
        :param y_adv:
        :return: True: y_true >= y_adv always holds, False: y_true >= y_adv MAY not hold
        """
        assert self.l is not None and self.u is not None
        assert len(self.l) == len(self.Ws)
        assert len(self.u) == len(self.bs)
        assert len(self.l) == len(self.u)
        assert len(self.Ws) == len(self.bs)

        l = self.l[-1]
        u = self.u[-1]
        l = np.maximum(l, 0)
        u = np.maximum(u, 0)
        W = self.Ws[-1]
        b = self.bs[-1]
        W_delta = W[y_true] - W[y_adv]
        b_delta = b[y_true] - b[y_adv]
        lb = np.dot(np.clip(W_delta, a_min=0., a_max=None), l) + np.dot(np.clip(W_delta, a_min=None, a_max=0.), u) + b_delta
        # print(l)
        # print(u)
        # print(u-l)
        # print(y_true, y_adv, lb)
        return lb >= 0.

    def calculate_bound(self, x0, eps):
        raise NotImplementedError("Haven't implemented yet.")


class IntervalBound(BoundCalculator):

    def calculate_bound(self, x0, eps):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.l = [np.clip(x0 - eps, a_min=self.in_min, a_max=self.in_max)]
        self.u = [np.clip(x0 + eps, a_min=self.in_min, a_max=self.in_max)]

        for i in range(len(self.Ws) - 1):
            now_l = self.l[-1]
            now_u = self.u[-1]
            if i > 0:
                now_l = np.clip(now_l, a_min=0., a_max=None)
                now_u = np.clip(now_u, a_min=0., a_max=None)
            W, b = self.Ws[i], self.bs[i]
            new_l = np.matmul(np.clip(W, a_min=0., a_max=None), now_l) + np.matmul(np.clip(W, a_min=None, a_max=0.), now_u) + b
            new_u = np.matmul(np.clip(W, a_min=None, a_max=0.), now_l) + np.matmul(np.clip(W, a_min=0., a_max=None), now_u) + b
            self.l.append(new_l)
            self.u.append(new_u)


class FastLinBound(BoundCalculator):

    def _form_diag(self, l, u):
        d = np.zeros(l.shape[0])
        for i in range(d.shape[0]):
            if u[i] >= 1e-6 and l[i] <= -1e-6:
                d[i] = u[i] / (u[i] - l[i])
            elif u[i] <= -1e-6:
                d[i] = 0.
            else:
                d[i] = 1.
        return np.diag(d)

    def calculate_bound(self, x0, eps):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.l = [np.clip(x0 - eps, a_min=self.in_min, a_max=self.in_max)]
        self.u = [np.clip(x0 + eps, a_min=self.in_min, a_max=self.in_max)]

        A0 = self.Ws[0]
        A = list()

        for i in range(len(self.Ws) - 1):
            T = [None for _ in range(i)]
            H = [None for _ in range(i)]
            for k in range(i - 1, -1, -1):
                if k == i - 1:
                    D = self._form_diag(self.l[-1], self.u[-1])
                    A.append(np.matmul(self.Ws[i], D))
                else:
                    A[k] = np.matmul(A[-1], A[k])
                T[k] = np.zeros_like(A[k].T)
                H[k] = np.zeros_like(A[k].T)
                for r in range(self.l[k+1].shape[0]):
                    if self.u[k+1][r] >= 1e-6 and self.l[k+1][r] <= -1e-6:
                        for j in range(A[k].shape[0]):
                            if A[k][j, r] > 0.:
                                T[k][r, j] = self.l[k+1][r]
                            else:
                                H[k][r, j] = self.l[k+1][r]
            if i > 0:
                A0 = np.matmul(A[-1], A0)
            nowl = list()
            nowu = list()
            for j in range(self.Ws[i].shape[0]):
                nu_j = np.dot(A0[j], x0) + self.bs[i][j]
                mu_p_j = mu_n_j = 0.
                for k in range(0, i):
                    mu_p_j -= np.dot(A[k][j], (T[k].T)[j])
                    mu_n_j -= np.dot(A[k][j], (H[k].T)[j])
                    nu_j += np.dot(A[k][j], self.bs[k])
                nowl.append(mu_n_j + nu_j - eps * np.sum(np.abs(A0[j])))
                nowu.append(mu_p_j + nu_j + eps * np.sum(np.abs(A0[j])))
            self.l.append(np.array(nowl))
            self.u.append(np.array(nowu))


class IntervalFastLinBound(BoundCalculator):

    def __init__(self, model, in_shape, in_min, in_max):
        super(IntervalFastLinBound, self).__init__(model, in_shape, in_min, in_max)

        self.interval_calc = IntervalBound(model, in_shape, in_min, in_max)
        self.fastlin_calc = FastLinBound(model, in_shape, in_min, in_max)

    def calculate_bound(self, x0, eps):
        self.interval_calc.calculate_bound(x0, eps)
        self.fastlin_calc.calculate_bound(x0, eps)

        self.l = list()
        self.u = list()
        for i in range(len(self.interval_calc.l)):
            self.l.append(np.maximum(self.interval_calc.l[i], self.fastlin_calc.l[i]))
            self.u.append(np.minimum(self.interval_calc.u[i], self.fastlin_calc.u[i]))

