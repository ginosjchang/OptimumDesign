from utils.Loss import loss_mse
import numpy as np
import math

#使用learning rate來調整參數
class GD:
    def __init__(self, model, loss_fn = loss_mse, lr=1e-2):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
    
    def one_epoch(self, X, Y):
        loss = self.loss_fn(self.model(X), Y)
        self.model.backward(Y)

        self.step()
        return loss

    def step(self):
        df = self.model.grad()

        self.model.set_param(self.model.parameters() - self.lr * df)

# 使用 Fletcher Reeves conjugate gradient 收斂參數
class FR:
    def __init__(self, model, loss_fn = loss_mse, lr=1e-2):
        self.model = model
        self.n = len(model)
        self.j = self.n + 1
        self.reset = 0
        self.loss_fn = loss_fn
        self.x0 = model.parameters()

    
    def one_epoch(self, X, Y):
        def forward_propagation_loss(param):
            self.model.set_param(param)
            return self.loss_fn(self.model(X), Y)

        def find_min(x1, v, h=0.1, cc=-1):
            a,b = bracket(x1,v,h=h)
            if np.sum(a) == 0.:
                return [False,None]
            opt = Gold_sec_nD(np.array([a,b]),cc=cc)
            return [True,opt]

        def bracket(x1, v, h=0.1):#from the book: Numerical methods in Engineering with Python
            c = 1.618033989
            f1 = forward_propagation_loss(x1)
            x2 = x1 + v*h
            f2 = forward_propagation_loss(x2)
            # Determine downhill direction and change sign of h if needed
            if f2 > f1:
                h = -h
                x2 = x1 + v*h
                f2 = forward_propagation_loss(x2)
                # Check if minimum between x1 - h and x1 + h
                if f2 > f1: 
                    return x2, x1- v*h
            # Search loop
            for i in range (100):
                h = c* h
                x3 = x2+ v*h
                f3 = forward_propagation_loss(x3)
                if f3 > f2:
                    return x1, x3
                x1 = x2
                x2 = x3
                f1 = f2
                f2 = f3
            print("The bracket did not include a minimum")
            return 0., 0.

        def Gold_sec_nD(intv: np.ndarray, cc=-1):
            '''
            use golden select method to find local minimum or maximum
            '''
            if cc == -1: # converge condition
                cc = 1.0e-4
            alf = (5**0.5 -1)/ 2 # golden ratio = 0.618... 
            a = intv[0]
            b = intv[1]
            lam= a+ (1- alf)* (b- a)  # |--------|-----|--------|
            mu = a+ alf* (b- a)       # a       lam   mu        b
            fl = forward_propagation_loss(lam)
            fm = forward_propagation_loss(mu) #At first, 2 function evaluations are needed
            iter = 0
            while float(math.dist(a,b)) > cc and iter < 500:
                iter +=1
                # n_iter = n_iter+1      
                if fl > fm:           # |--------|-----|--------|
                    a = lam           # x     a(lam)  lam(mu)   b
                    lam = mu
                    fl = fm
                    mu = a+ alf* (b- a)
                    fm = forward_propagation_loss(mu)   #In the while loop, only 1 more function evalution is needed
                    # optv.append(fl)
                else:
                    b = mu
                    mu = lam
                    fm = fl
                    lam = a+ (1- alf)* (b- a)
                    fl = forward_propagation_loss(lam)
                    # optv.append(fm)
            if forward_propagation_loss(a) < forward_propagation_loss(b): #compare f(a) with f(b), and xopt is the one with smaller func value
                xopt = a
            else:
                xopt = b
                
            return xopt

        def backward_propagation_gradient(param):
            self.model.set_param(param)
            loss = self.loss_fn(self.model(X), Y)
            self.model.backward(Y)
            return self.model.grad()

        if self.j <= self.n:
            cond, x1 = find_min(self.x0, self.s0)
            if not cond:
                self.x0 = self.model.reset().parameters()
                self.reset += 1
                return forward_propagation_loss(self.x0)
            df1 = backward_propagation_gradient(x1)
            s1 = -1*df1 + (np.inner(df1, df1)/np.inner(self.df0, self.df0)) * self.s0
            self.s0 = s1
            # if math.dist(self.x0, x1) < 1e-4:
            #     self.x0 = self.model.reset().parameters()
            #     self.reset += 1
            #     return forward_propagation_loss(self.x0)
            self.x0 = x1
            self.df0 = df1
            self.j += 1
        else:
            self.df0 = backward_propagation_gradient(self.x0)
            self.s0 = -1 * self.df0
            self.j = 1

        self.model.set_param(self.x0)
        return forward_propagation_loss(self.x0)
    
class Nelder_Mead:
    def __init__(self, model, loss_fn = loss_mse, lr=1e-2):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

        start_points = []
        for _ in range(len(self.model) + 1):
            start_points.append(self.model.reset(True).parameters())
        self.p = np.array(start_points)

    def one_epoch(self, X, Y):
        def forward_propagation_loss(param):
            self.model.set_param(param)
            return self.loss_fn(self.model(X), Y)
        
        self.p_v = [[np.array(p,dtype=np.float64),forward_propagation_loss(p)] for p in self.p]
        self.p_v.sort(key=lambda x: x[1], reverse=True)
        self.p = np.array(self.p_v,dtype=object)[:,0]

        pa, va = self.p_v[0]
        pb, vb = self.p_v[-2]
        pc, vc = self.p_v[-1]
        centers = np.average(self.p)

        pavg = np.average(self.p[1:])
        pr = pavg + 1*(pavg - pa)
        vr = forward_propagation_loss(pr)
        if vc > vr:
            pe = pavg + 2*(pr - pavg)
            ve = forward_propagation_loss(pe)
            if vr > ve:
                self.p_v[0] = [pe,ve]
            else:
                self.p_v[0] = [pr,vr]
        else:
            if vb >= vr:
                self.p_v[0] = [pr,vr]
            else:
                if vr < va:
                    pp = pr
                else:
                    pp = pa
                pct = pavg + 0.5*(pp - pavg)
                if forward_propagation_loss(pct) > forward_propagation_loss(pp):
                    for j in range(len(self.p)-1):
                        self.p[j] += (pc-self.p[j])/2
                        self.p_v[j][1] = forward_propagation_loss(self.p[j])
                else:
                    self.p_v[0] = [pct,forward_propagation_loss(pct)]
        
        self.p_v.sort(key=lambda x: x[1], reverse=True)
        self.p = np.array(self.p_v,dtype=object)[:,0]

        param, loss = self.p_v[-1]

        self.model.set_param(param)
        return loss

class DFP:
    def __init__(self, model, loss_fn = loss_mse, lr=1e-2):
        self.model = model
        self.loss_fn = loss_fn
        self.B = np.identity(len(model))
        self.niter = 1
        self.x0 = None

    def one_epoch(self, X, Y):
        def forward_propagation_loss(param):
            self.model.set_param(param)
            return self.loss_fn(self.model(X), Y)

        def find_min(x1, v, h=0.1, cc=-1):
            a,b = bracket(x1,v,h=h)
            if np.sum(a) == 0.:
                return [False,None]
            opt = Gold_sec_nD(np.array([a,b]),cc=cc)
            return [True,opt]

        def bracket(x1, v, h=0.1):#from the book: Numerical methods in Engineering with Python
            c = 1.618033989
            f1 = forward_propagation_loss(x1)
            x2 = x1 + v*h
            f2 = forward_propagation_loss(x2)
            # Determine downhill direction and change sign of h if needed
            if f2 > f1:
                h = -h
                x2 = x1 + v*h
                f2 = forward_propagation_loss(x2)
                # Check if minimum between x1 - h and x1 + h
                if f2 > f1: 
                    return x2, x1- v*h
            # Search loop
            for i in range (100):
                h = c* h
                x3 = x2+ v*h
                f3 = forward_propagation_loss(x3)
                if f3 > f2:
                    return x1, x3
                x1 = x2
                x2 = x3
                f1 = f2
                f2 = f3
            print("The bracket did not include a minimum")
            return 0., 0.

        def Gold_sec_nD(intv: np.ndarray, cc=-1):
            '''
            use golden select method to find local minimum or maximum
            '''
            if cc == -1: # converge condition
                cc = 1.0e-4
            alf = (5**0.5 -1)/ 2 # golden ratio = 0.618... 
            a = intv[0]
            b = intv[1]
            lam= a+ (1- alf)* (b- a)  # |--------|-----|--------|
            mu = a+ alf* (b- a)       # a       lam   mu        b
            fl = forward_propagation_loss(lam)
            fm = forward_propagation_loss(mu) #At first, 2 function evaluations are needed
            iter = 0
            while float(math.dist(a,b)) > cc and iter < 500:
                iter +=1
                # n_iter = n_iter+1      
                if fl > fm:           # |--------|-----|--------|
                    a = lam           # x     a(lam)  lam(mu)   b
                    lam = mu
                    fl = fm
                    mu = a+ alf* (b- a)
                    fm = forward_propagation_loss(mu)   #In the while loop, only 1 more function evalution is needed
                    # optv.append(fl)
                else:
                    b = mu
                    mu = lam
                    fm = fl
                    lam = a+ (1- alf)* (b- a)
                    fl = forward_propagation_loss(lam)
                    # optv.append(fm)
            if forward_propagation_loss(a) < forward_propagation_loss(b): #compare f(a) with f(b), and xopt is the one with smaller func value
                xopt = a
            else:
                xopt = b
                
            return xopt

        def backward_propagation_gradient(param):
            self.model.set_param(param)
            loss = self.loss_fn(self.model(X), Y)
            self.model.backward(Y)
            return self.model.grad()

        def matrix_multiply_2D(a, b):
            if len(a.shape) == 1:
                a = a.reshape(1,-1)
            if len(b.shape) == 1:
                b = b.reshape(-1,1)
            res = np.zeros((a.shape[0], b.shape[1]))
            for i in range(a.shape[0]):
                for j in range(b.shape[1]):
                    for k in range(a.shape[1]):
                        res[i][j] += a[i][k]* b[k][j]
            if res.shape[1] == 1:
                res = res.reshape(-1)
            return res

        x0 = self.model.parameters()

        df0 = backward_propagation_gradient(x0)
        s = -1* np.matmul(self.B, df0)
        _, x1 = find_min(x0, s)
        df1 = backward_propagation_gradient(x1)

        lam = np.average((x1-x0)/s)
        g = df1 - df0
        M = lam * matrix_multiply_2D(s.reshape(-1,1),s.reshape(-1,1)) / np.inner(s,g)
        Bg = np.matmul(self.B,g)
        N = -1 * matrix_multiply_2D(Bg.reshape(-1,1),Bg.reshape(-1,1)) / np.inner(g,Bg)
        self.B = self.B + M + N
        
        loss = forward_propagation_loss(x0)
        self.model.set_param(x1)
        self.niter += 1
        if self.niter % 10 == 0:
            self.B = np.identity(len(self.model))
        return loss