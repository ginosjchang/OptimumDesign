# import Python Libraries
import numpy as np
from matplotlib import pyplot as plt
import math

# Sigmoid Function
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

# Relu Function
def relu(Z):
    return np.where(Z>0, Z, 0)

# Leaky Relu Function
def leaky_relu(Z):
    return Z * np.where(Z>0, 1, 0.1)

# Derivative of the Sigmoid function
def d_sigmoid(Z):
    return sigmoid(Z) * (1-sigmoid(Z))

# Derivative of the Relu Function
def d_relu(Z):
    return np.where(Z>0, 1, 0)

# Derivative of the Leaky Relu Function
def d_leaky_relu(Z):
    return np.where(Z>0, 1, 0.1)

# Loss function: (MSE)
    # input: Model output: A2 & groundtruth: Y
    # output: Loss value

def loss_mse(A2, Y):
    loss = (A2 - Y)**2
    loss = loss.mean()

    return loss

# Derivative of the Loss function
def d_loss_mse(A2, Y):
    dsq_diffs = (2 * (A2 - Y))/m
    
    return dsq_diffs

def loss_ce(A2, Y):
    loss = np.where(Y > 0.5, np.log(A2), np.log(1-A2))
    return -1*loss.mean()

def d_loss_ce(A2, Y):
    d = np.where(Y > 0.5, -1/A2, 1/(1-A2))
    return d

# 初始化參數
    # input: 每層的單元數
    # output: parameters

# Weights: W1 & W2 從正態分布中隨機初始化
# Bias: B1 & B2 初始化為 0

def initialize_parameters_FR(n_x, n_h, n_y):
    # return np.random.randn(n_h*(n_x+n_y+1)+n_y)
    w1 = np.random.randn(n_h*n_x)
    b1 = np.zeros(n_h)
    w2 = np.random.randn(n_y*n_h)
    b2 = np.zeros(n_y)
    return np.concatenate((w1,b1,w2,b2))

def initialize_parameters_NM(n_x, n_h, n_y):
    n = n_h*(n_x+n_y+1) + n_y + 1
    params = []
    for i in range(n):
        # w1 = np.random.randn(n_h*n_x)
        # w2 = np.random.randn(n_y*n_h)
        # b1 = np.random.randn(n_h)
        # b2 = np.random.randn(n_y)
        params.append(np.random.randn(n_h*(n_x+n_y+1)+n_y))
    return params

def initialize_parameters(n_x, n_h, n_y):
    # 初始化參數
    W1 = np.random.randn(n_h, n_x)
    B1 = np.zeros((n_h, 1))
    # B1 = np.random.randn(n_h, 1)
    W2 = np.random.randn(n_y, n_h)
    B2 = np.zeros((n_y, 1))
    # B2 = np.random.randn(n_y, 1)
    
    # 寫入parameters字典
    parameters = { 
        "W1": W1,
        "B1": B1,
        "W2": W2,
        "B2": B2
    }
    # print("init:\n",parameters)
    return parameters


# Forward Propagation
    # input: Model input: X & parameters
    # output: Model output: Y & cache(稍後在back_prop中使用)

def forward_prop(X, parameters):
    # 載入參數
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    B1 = parameters["B1"]
    B2 = parameters["B2"]

    # Forward propagation
    Z1 = np.dot(W1, X) + B1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
 
    # 寫入cache字典
    cache = {
        "A1": A1,
        "A2": A2,
        "Z1": Z1,
        "Z2": Z2,
        "W1": W1,
        "B1": B1,
        "W2": W2,
        "B2": B2
    }

    return A2, cache

# Backward Propagation
    # input: Model input: X & groundtruth: Y & cache
    # output: gradient

def backward_prop(X, Y, cache, loss_diff):
    # 載入cache
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    W2 = cache["W2"]

    # Chain rule
    dZ2 = np.multiply(d_loss_ce(A2, Y), d_sigmoid(Z2))
    dW2 = np.dot(dZ2, A1.T)
    dB2 = np.sum(dZ2, axis = 1, keepdims = True)
     
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, d_leaky_relu(Z1))
    dW1 = np.dot(dZ1, X.T)
    dB1 = np.sum(dZ1, axis = 1, keepdims = True)

    # perturbation
    ptb = abs(loss_diff) > 1000
    # 寫入gradient字典
    gradients = {
        "dW1": dW1 + ptb * np.random.randn(*dW1.shape)/5,
        "dW2": dW2 + ptb * np.random.randn(*dW2.shape)/5,
        "dB1": dB1 + ptb * np.random.randn(*dB1.shape)/5,
        "dB2": dB2 + ptb * np.random.randn(*dB2.shape)/5,
    }
        
    return gradients


# 使用 Fletcher Reeves conjugate gradient 收斂參數
def Fletcher_Reeves_conjugate_gradient(X, Y, n_x, n_h, n_y, loss_fn=loss_mse, cc=1e-2):

    param = initialize_parameters_FR(n_x, n_h, n_y)
    n = len(param)
    j = 1
    x0 = param
    iter = 0
    reset = 0

    def arr2dict(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        parameters = { 
            "W1": w1,
            "B1": b1,
            "W2": w2,
            "B2": b2
        }
        return parameters

    def forward_propagation_loss(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        z1 = np.dot(w1, X) + b1
        a1 = leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        return loss_fn(a2, Y)

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
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        z1 = np.dot(w1, X) + b1
        a1 = leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        
        losses.append(loss_fn(a2,Y))
        dz2 = np.multiply(d_loss_ce(a2, Y), d_sigmoid(z2))
        dw2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis = 1, keepdims = True)
        da1 = np.dot(w2.T, dz2)
        dz1 = np.multiply(da1, d_leaky_relu(z1))
        dw1 = np.dot(dz1, X.T)
        db1 = np.sum(dz1, axis = 1, keepdims = True)
        gradient = np.concatenate((dw1.reshape(-1),
                                   db1.reshape(-1),
                                   dw2.reshape(-1),
                                   db2.reshape(-1),))
        return gradient

    while True:
        df0 = backward_propagation_gradient(x0)
        s0 = -1 * df0
        while j <= n:
            if math.dist(s0, np.zeros_like(s0)) < cc and forward_propagation_loss(x0) < cc:
                return reset, arr2dict(x0)
            # find lambda to make x+l*s is minimum in direction of s.
            ## use hw2 method
            ## a, b = bracket(f, x0, s0, h=0.1)
            ## x1 = Gold_sec_nD(f, np.array([a,b]))
            cond, x1 = find_min(x0, s0)
            if not cond:
                x0 = initialize_parameters_FR(n_x, n_h, n_y)
                reset += 1
                break
            df1 = backward_propagation_gradient(x1)
            s1 = -1*df1 + (np.inner(df1, df1)/np.inner(df0, df0)) * s0
            s0 = s1
            if math.dist(x0,x1) < 1e-4:
                x0 = initialize_parameters_FR(n_x, n_h, n_y)
                reset += 1
                break
            x0 = x1
            df0 = df1
            j += 1
            iter +=1
        j = 1
        iter +=1

# 使用 Nelder Mead Downhill simplex method 收斂參數
def Nelder_Mead_method(X, Y, n_x, n_h, n_y, loss_fn=loss_mse, cc=1e-4):
    
    def arr2dict(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        parameters = { 
            "W1": w1,
            "B1": b1,
            "W2": w2,
            "B2": b2
        }
        return parameters

    def forward_propagation_loss(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        z1 = np.dot(w1, X) + b1
        a1 = leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        return loss_fn(a2, Y)

    start_points = initialize_parameters_NM(n_x, n_h, n_y)
    p_v = [[np.array(p,dtype=np.float64),forward_propagation_loss(p)] for p in start_points]
    p_v.sort(key=lambda x: x[1], reverse=True)
    p = np.array(p_v,dtype=object)[:,0]
    # v = np.array(p_v,dtype=object)[:,1]
    pa,va = p_v[0]
    pb,vb = p_v[-2]
    pc,vc = p_v[-1]
    centers = np.average(p)
    niter = 0
    while math.dist(pa,pc) > cc and niter < 1000:
        pavg = np.average(p[1:])
        pr = pavg + 1*(pavg - pa)
        vr = forward_propagation_loss(pr)
        if vc > vr:
            pe = pavg + 2*(pr - pavg)
            ve = forward_propagation_loss(pe)
            if vr > ve:
                p_v[0] = [pe,ve]
            else:
                p_v[0] = [pr,vr]
        else:
            if vb >= vr:
                p_v[0] = [pr,vr]
            else:
                if vr < va:
                    pp = pr
                else:
                    pp = pa
                pct = pavg + 0.5*(pp - pavg)
                if forward_propagation_loss(pct) > forward_propagation_loss(pp):
                    for j in range(len(p)-1):
                        p[j] += (pc-p[j])/2
                        p_v[j][1] = forward_propagation_loss(p[j])
                else:
                    p_v[0] = [pct,forward_propagation_loss(pct)]
        p_v.sort(key=lambda x: x[1], reverse=True)
        pa,va = p_v[0]
        pb,vb = p_v[1]
        pc,vc = p_v[-1]
        losses.append(vc)
        if niter%1000 == 0:
            print(vc)
        centers = np.average(p)
        niter += 1
    print(f'iteration: {niter}')

    return p_v, arr2dict(pc)

# 使用 powell's conjugate direction method 收斂參數
def powell_conjugate(X, Y, n_x, n_h, n_y, loss_fn=loss_mse):
    
    def arr2dict(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        parameters = { 
            "W1": w1,
            "B1": b1,
            "W2": w2,
            "B2": b2
        }
        return parameters

    def forward_propagation_loss(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        z1 = np.dot(w1, X) + b1
        a1 = leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        return loss_fn(a2, Y)

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
    
    param = initialize_parameters_FR(n_x, n_h, n_y)
    f1 = forward_propagation_loss(param)
    d = np.identity(len(param))
    niter = 0
    while niter < 100:
        z = [param]
        y = [param]
        x = param
        for j in range(len(param)):
            for i in range(len(param)):
                _, param = find_min(param, d[i])
                z.append(param)
            _, param = find_min(param, z[-1]-z[0])
            losses.append(forward_propagation_loss(param))
            y.append(param)
            if j == len(param) -1:
                continue
            np.delete(d,0)
            np.concatenate((d,np.array([z[-1]-z[0]])))
            z = [y[j+1]]
        if math.dist(y[-1],x) < 1e-4 and losses[-1] < 1e-2:
            break
        else:
            param = y[-1]
        niter += 1
    return niter, arr2dict(x)

# 使用 powell's conjugate direction method 收斂參數
def gradient_descent_Newton_method(X, Y, n_x, n_h, n_y, loss_fn=loss_mse):
    
    def arr2dict(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        parameters = { 
            "W1": w1,
            "B1": b1,
            "W2": w2,
            "B2": b2
        }
        return parameters

    def forward_propagation_loss(param):
        w1 = param[:n_h*n_x].reshape(n_h,n_x)
        b1 = param[n_h*n_x:n_h*(n_x+1)].reshape(n_h,1)
        w2 = param[n_h*(n_x+1):n_h*((n_x+1)+n_y)].reshape(n_y,n_h)
        b2 = param[n_h*((n_x+1)+n_y):].reshape(n_y,1)
        z1 = np.dot(w1, X) + b1
        a1 = leaky_relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        return loss_fn(a2, Y)

    def find_armijo_min(x1, v, h=0.1, cc=-1):
        lam = Armijo(loss_fn, x1, v, h)
        return lam

    def gradient_nD(f, x, h=1e-3):
        '''
        Compute the gradient of f at one given point x
            f: nD function
            x: point location which degree > 1
            h: interval, not important
        '''
        gradient = []
        for i in range(len(x)):
            h_v = np.array([h if i==j else 0 for j in range(len(x))])
            x_plus = f(*(np.array(x) + h_v))
            x_minus= f(*(np.array(x) - h_v))
            gradient.append((x_plus - x_minus) / (2*h))
        return np.array(gradient)

    def Armijo(f, x, s, lam):
        epsilon = 0.2
        if (f(x+ lam*s) <= f(x) + lam*epsilon*np.inner(gradient_nD(f, x),x)):
            while (f(x+ lam*s) <= f(x) + lam*epsilon*np.inner(gradient_nD(f, x),x)):
                lam = lam*2
        else:
            while (f(x+ lam*s) > f(x) + lam*epsilon*np.inner(gradient_nD(f, x),x)):
                lam = lam/2
        return x+ lam*s
    


# 定義訓練迴圈

def training_loop(X, Y, n_x, n_h, n_y, n_epoch, learning_rate):
    # 更新參數

    # 使用Gradient Descent
        # 輸入: parameter & gradient & learning rate
        # 輸出: parameter
    def update_parameters(parameters, gradients, learningRate):
        
        parameters["W1"] = parameters["W1"] - learningRate * gradients["dW1"]
        parameters["W2"] = parameters["W2"] - learningRate * gradients["dW2"]
        parameters["B1"] = parameters["B1"] - learningRate * gradients["dB1"]
        parameters["B2"] = parameters["B2"] - learningRate * gradients["dB2"]
        
        return parameters
    
    # 初始化參數
    parameters = initialize_parameters(n_x, n_h, n_y)

    for epoch in range(0, n_epoch + 1):
        # Forward propagation
        A2, cache = forward_prop(X, parameters)

        # 計算Loss
        loss = loss_ce(A2, Y)
        # 記錄Loss值(輸出圖片用)
        losses.append(loss)

        # 計算gradient
        grads = backward_prop(X, Y, cache, losses[epoch]/(losses[epoch]-losses[epoch-1]))

        # 更新參數
        parameters = update_parameters(parameters, grads, learning_rate)

        # 追蹤Loss變化
        if(epoch % 10000 == 0):
            print("Epoch {:d}, Loss {:f}".format(epoch, loss))

        # learning rate decay
        # if epoch % 1000 == 999:
        #     learning_rate *= 0.9
    # print("---Training Finished---\n")
    # print('\nend:\n',parameters)
    return parameters


# 預測測試資料
    # input: X & parameters (for forward_prop)
    # output: 預測結果: y_predict (0 or 1)

def predict(X, parameters):
    A2, _ = forward_prop(X, parameters)
    A2 = np.squeeze(A2) 

    return A2 >= 0.5, A2

if __name__ =='__main__':
    # The 4 training examples by columns
    X = np.array([[0, 0, 1, 1], 
                [0, 1, 0, 1]])

    # The outputs of the XOR for every example in X
    Y = np.array([[0, 1, 1, 0]])

    # Set the hyperparameters
    n_x = X.shape[0]        # No. of neurons in first layer
    n_h = 2                 # No. of neurons in hidden layer
    n_y = Y.shape[0]        # No. of neurons in output layer
    m = X.shape[1]          # No. of training examples
    n_epochs = 50000
    learning_rate = 0.1
    losses = [] # 記錄Loss值(輸出圖片用)

    ## Main training process
    ## orignal gradient descent
    trained_parameters = training_loop(
        X,
        Y,
        n_x,
        n_h,
        n_y,
        n_epochs,
        learning_rate)
    # reset_times, trained_parameters = Fletcher_Reeves_conjugate_gradient(
    #     X,
    #     Y,
    #     n_x,
    #     n_h,
    #     n_y,
    #     loss_mse,
    # )
    # pv_set, trained_parameters = Nelder_Mead_method(
    #     X,
    #     Y,
    #     n_x,
    #     n_h,
    #     n_y,
    #     loss_mse,
    # )
    # niter, trained_parameters = powell_conjugate(
    #     X,
    #     Y,
    #     n_x,
    #     n_h,
    #     n_y,
    #     loss_mse,
    # )

    # print(f"reset {reset_times} times")
    # 4種組合之測試data
    test_1 = np.array([[0], [0]])
    test_2 = np.array([[0], [1]])
    test_3 = np.array([[1], [0]])
    test_4 = np.array([[1], [1]])

    # 4種組合之預測結果
    predict_1, origin_1 = predict(test_1, trained_parameters)
    predict_2, origin_2 = predict(test_2, trained_parameters)
    predict_3, origin_3 = predict(test_3, trained_parameters)
    predict_4, origin_4 = predict(test_4, trained_parameters)

    # Print the result
    print('---Neural Network Prediction Test---')
    print('({:d}, {:d}) is {:d},   Model output: {:f}'.format( test_1[0][0], test_1[1][0], predict_1, origin_1))
    print('({:d}, {:d}) is {:d},   Model output: {:f}'.format( test_2[0][0], test_2[1][0], predict_2, origin_2))
    print('({:d}, {:d}) is {:d},   Model output: {:f}'.format( test_3[0][0], test_3[1][0], predict_3, origin_3))
    print('({:d}, {:d}) is {:d},   Model output: {:f}'.format( test_4[0][0], test_4[1][0], predict_4, origin_4))


    # Evaluating the performance
    plt.figure(num = "Loss - Epochs")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.show()