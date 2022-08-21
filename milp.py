
import torch
from torch import nn

import numpy as np
import cvxpy as cp


class MILPVerifier_dnn:

    def __init__(self, model, in_shape, in_min, in_max):
        super(MILPVerifier_dnn, self).__init__()

        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()
        in_numel = in_shape
        shapes.append(in_shape)
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

    def construct(self, l, u, x0, eps, norm, fix0 = 0, provar = False):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()
        self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)
        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        if norm == 0:
            self.z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(self.z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(self.z,eps)))
            self.obj = cp.Minimize(cp.norm1(self.z))

        if norm == 1:
            self.obj = cp.Minimize(cp.norm1(self.cx - x0))
        if norm == 2:
            self.obj = cp.Minimize(cp.norm2(self.cx - x0))
        if norm == -1:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))
        if fix0:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.constraints.append(cp.norm1(z) <= fix0)
            # self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))
        if provar is not False:
            self.constraints.append((self.cx[provar] == x0[provar] ))
            # self.constraints.append((provar @ (self.cx - x0 ) == 0))

        pre = self.cx
        for i in range(len(self.Ws) - 1):
            now_x = (self.Ws[i] @ pre) + self.bs[i]
            now_shape = self.shapes[i + 1]
            now_y = cp.Variable(now_shape)
            now_a = cp.Variable(now_shape, boolean=True)
            for j in range(now_shape):
                if l[i + 1][j] >= 0:
                    self.constraints.extend([now_y[j] == now_x[j]])
                elif u[i + 1][j] <= 0:
                    self.constraints.extend([now_y[j] == 0.])
                else:
                    self.constraints.extend([
                        (now_y[j] <= now_x[j] - (1 - now_a[j]) * l[i + 1][j]),
                        (now_y[j] >= now_x[j]),
                        (now_y[j] <= now_a[j] * u[i + 1][j]),
                        (now_y[j] >= 0.)
                    ])
            # self.constraints.extend([(now_y <= now_x - cp.multiply((1 - now_a), l[i + 1])), (now_y >= now_x), (now_y <= cp.multiply(now_a, u[i + 1])), (now_y >= 0)])
            pre = now_y
        self.last_x = pre

    def prepare_verify(self, y0):
        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        # last_w @ x0 +last_b
        mask = np.ones(last_w.shape[0], dtype=bool)
        mask[y0] = False
        last_w_masked = last_w[mask,:]
        last_b_masked = last_b[mask]

        # Big-M converts max function
        output = last_w_masked @ self.last_x + last_b_masked
        self.zm = cp.Variable(last_b_masked.shape[0],boolean=True)
        self.max = cp.Variable()
        self.constraints.append((cp.sum(self.zm) == 1))
        self.constraints.append((self.max >= output))
        self.constraints.append((self.max <= output + cp.multiply((1 - self.zm),1e6)))

        self.constraints.extend([self.max  >= last_w[y0] @ self.last_x + last_b[y0] + 1e-8])
        self.prob = cp.Problem(self.obj, self.constraints)


class MILPVerifier_svm:

    def __init__(self, model, in_shape, in_min, in_max):
        super(MILPVerifier_svm, self).__init__()

        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()
        in_numel = in_shape
        shapes.append(in_shape)
        Ws.append(model[0])
        bs.append(model[1])
        shapes.append(len(model[1]))

        self.in_min, self.in_max = in_min, in_max

        self.in_numel = in_numel
        self.num_layers = num_layers
        self.shapes = shapes
        self.Ws = Ws
        self.bs = bs

    def construct(self, x0, eps, norm, fix0 = 0):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()
        self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)

        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        if norm == 0:
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.obj = cp.Minimize(cp.norm1(z))
        if norm == 1:
            self.obj = cp.Minimize(cp.norm1(self.cx - x0))
        if norm == 2:
            self.obj = cp.Minimize(cp.norm2(self.cx - x0))
        if norm == -1:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))

        if fix0:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.constraints.append(cp.norm1(z) <= fix0)

        self.last_x  = self.cx


    def prepare_verify(self, y0):
        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        # last_w @ x0 +last_b
        mask = np.ones(last_w.shape[0], dtype=bool)
        mask[y0] = False
        last_w_masked = last_w[mask,:]
        last_b_masked = last_b[mask]

        # Big-M converts max function
        output = last_w_masked @ self.last_x + last_b_masked
        self.zm = cp.Variable(last_b_masked.shape[0],boolean=True)
        self.max = cp.Variable()
        self.constraints.append((cp.sum(self.zm) == 1))
        self.constraints.append((self.max >= output))
        self.constraints.append((self.max <= output + cp.multiply((1 - self.zm),300)))

        self.constraints.extend([self.max  >= last_w[y0] @ self.last_x + last_b[y0]])
        self.prob = cp.Problem(self.obj, self.constraints)



class MILPVerifier_pca:

    def __init__(self, model, in_shape, in_min, in_max):
        super(MILPVerifier_pca, self).__init__()

        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()
        Ws = list()

        in_numel = in_shape
        shapes.append(in_shape)
        Ws.append(model[0])
        self.in_min, self.in_max = in_min, in_max

        self.in_numel = in_numel
        self.num_layers = num_layers
        self.shapes = shapes
        self.Ws = Ws
        self.mean = model[2]
        self.std = model[3]
        self.lim = model[1]

    def construct(self, x0, eps, norm, fix0 = 0):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()
        self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)

        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        if norm == 0:
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.obj = cp.Minimize(cp.norm1(z))
        if norm == 1:
            self.obj = cp.Minimize(cp.norm1(self.cx - x0))
        if norm == 2:
            self.obj = cp.Minimize(cp.norm2(self.cx - x0))
        if norm == -1:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))

        if fix0:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.constraints.append(cp.norm1(z) <= fix0)
            # self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))

        self.last_x  = self.cx


    def prepare_verify(self, y):
        last_w = self.Ws[-1]
        nor_x = (self.last_x - self.mean) / self.std
        e = nor_x @ last_w
        if y == 1:
            self.constraints.extend([cp.sum_squares(e) <= self.lim])
        if y == 0:
            self.constraints.extend([cp.sum_squares(e) >= self.lim])
        self.prob = cp.Problem(self.obj, self.constraints)

class MILPVerifier_ae:

    def __init__(self, model, in_shape, in_min, in_max):
        super(MILPVerifier_ae, self).__init__()
        self.lim = model[1]
        model = model[0]
        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()
        in_numel = in_shape
        shapes.append(in_shape)
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

    def construct(self, l, u, x0, eps, norm, fix0 = 0, provar = False):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()
        self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)
        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        if norm == 0:
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.obj = cp.Minimize(cp.norm1(z))

        if norm == 1:
            self.obj = cp.Minimize(cp.norm1(self.cx - x0))
        if norm == 2:
            self.obj = cp.Minimize(cp.norm2(self.cx - x0))
        if norm == -1:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))
        if fix0:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.constraints.append(cp.norm1(z) <= fix0)
            self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))


        pre = self.cx
        for i in range(len(self.Ws) - 1):
            now_x = (self.Ws[i] @ pre) + self.bs[i]
            now_shape = self.shapes[i + 1]
            now_y = cp.Variable(now_shape)
            now_a = cp.Variable(now_shape, boolean=True)
            for j in range(now_shape):
                if l[i + 1][j] >= 0:
                    self.constraints.extend([now_y[j] == now_x[j]])
                elif u[i + 1][j] <= 0:
                    self.constraints.extend([now_y[j] == 0.])
                else:
                    self.constraints.extend([
                        (now_y[j] <= now_x[j] - (1 - now_a[j]) * l[i + 1][j]),
                        (now_y[j] >= now_x[j]),
                        (now_y[j] <= now_a[j] * u[i + 1][j]),
                        (now_y[j] >= 0.)
                    ])
            # self.constraints.extend([(now_y <= now_x - cp.multiply((1 - now_a), l[i + 1])), (now_y >= now_x), (now_y <= cp.multiply(now_a, u[i + 1])), (now_y >= 0)])
            pre = now_y
        self.last_x = pre

    def prepare_verify(self, y):
        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        x_hat = (last_w @ self.last_x) + last_b

        if y == 1:
            self.constraints.extend([cp.sum_squares(x_hat - self.cx) <= self.lim])
        if y == 0:
            self.constraints.extend([cp.sum_squares(x_hat - self.cx) >= self.lim])
            # self.constraints.extend([cp.sum_squares(x_hat - self.cx) <= 1.0])

        self.prob = cp.Problem(self.obj, self.constraints)

class MILPVerifier_ae_vy:

    def __init__(self, model, in_shape, in_min, in_max):
        super(MILPVerifier_ae_vy, self).__init__()
        self.lim = model[1]
        model = model[0]
        in_numel = None
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()
        in_numel = in_shape
        shapes.append(in_shape)
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

    def construct(self, l, u, x0, eps, neps, norm):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()
        self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)
        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        if norm == 0:
            z = cp.Variable(self.in_numel,boolean=True)
            self.constraints.append((self.cx - x0 >=  cp.multiply(z,-eps)))
            self.constraints.append((self.cx - x0 <=  cp.multiply(z,eps)))
            self.constraints.append((cp.norm1(z) <= neps))

        if norm == 1:
            self.constraints.append((cp.norm1(self.cx - x0) <= neps))
        if norm == 2:
            self.constraints.append((cp.norm2(self.cx - x0) <= neps))
        if norm == -1:
            # aux_eps=cp.Variable()
            # self.constraints.append((self.cx - x0 <= aux_eps))
            # self.constraints.append((x0 - self.cx <= aux_eps))
            # self.obj = cp.Minimize(aux_eps)
            self.constraints.append((cp.norm_inf(self.cx - x0) <= neps))


        pre = self.cx
        for i in range(len(self.Ws) - 1):
            now_x = (self.Ws[i] @ pre) + self.bs[i]
            now_shape = self.shapes[i + 1]
            now_y = cp.Variable(now_shape)
            now_a = cp.Variable(now_shape, boolean=True)
            for j in range(now_shape):
                if l[i + 1][j] >= 0:
                    self.constraints.extend([now_y[j] == now_x[j]])
                elif u[i + 1][j] <= 0:
                    self.constraints.extend([now_y[j] == 0.])
                else:
                    self.constraints.extend([
                        (now_y[j] <= now_x[j] - (1 - now_a[j]) * l[i + 1][j]),
                        (now_y[j] >= now_x[j]),
                        (now_y[j] <= now_a[j] * u[i + 1][j]),
                        (now_y[j] >= 0.)
                    ])
            # self.constraints.extend([(now_y <= now_x - cp.multiply((1 - now_a), l[i + 1])), (now_y >= now_x), (now_y <= cp.multiply(now_a, u[i + 1])), (now_y >= 0)])
            pre = now_y
        self.last_x = pre

    def prepare_verify(self, y0, yp):
        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        x_hat = (last_w @ self.last_x) + last_b

        if yp == 0:
            self.obj= cp.Minimize( cp.sum_squares(x_hat - self.cx) - self.lim)
        if yp == 1:
            self.obj= cp.Minimize( self.lim - cp.sum_squares(x_hat - self.cx))
            # self.constraints.extend([cp.sum_squares(x_hat - self.cx) <= 1.0])

        self.prob = cp.Problem(self.obj, self.constraints)