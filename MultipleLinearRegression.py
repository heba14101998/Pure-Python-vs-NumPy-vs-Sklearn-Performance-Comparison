import numpy as np
import itertools
class LinearRegression:
    def __init__(self, X, Y):
        x0=np.ones((X.shape[0],1))
        X=np.append(x0,X,axis=1)
        self.X = X
        self.Y = Y
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.theta = np.random.randn(self.n) 
        self.theta = self.theta.reshape(-1,self.theta.ndim)
        self.J = 0.0
    
    def gradient_descent(self, lr,n_epochs):
        self.cost_history = []
        self.theta_history= []
        
        for _ in itertools.repeat(None, n_epochs):
            
            # computed value of Y
            Y_hat = np.dot(self.X, self.theta)
            # the diffrance between the exact and the computed value
            tmp = Y_hat - self.Y
            # cost function
            self.J = (1/self.m )* np.sum(tmp**2)
            self.cost_history.append(self.J)
            self.theta_history.append(self.theta)
            # gradient of the cost function
            d_mse = (2/self.m )* self.X.T.dot(tmp)
            self.theta -= lr * d_mse
        
        return self.theta, self.theta_history, self.cost_history

    def predict(self, X_test, Y_test):
        
        x0=np.ones((X.shape[0],1))
        X_test=np.append(x0,X_test,axis=1)
        self.Y_pred =  np.dot(self.theta.T, self.X_test)
        self.err_mse = (1/self.m )* np.sum( (Y_test - self.Y_pred)**2 )
        self.err_percent=(abs(self.Y_pred-Y_test)/Y_test)*100
        
        return self.Y_pred, self.err_mse, self.err_percent
        
    def get_weights(self):
        return self.theta
    def get_history(self):
        return self.theta_history, self.cost_history
    def get_pred_values(self):
        return self.Y_pred
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.Y
    
class FeatureScaler:
    
    def __init__(self, X, Y):
        self.X = X.copy()
        Y = Y.reshape(X.shape[0],1)
        self.Y = Y.copy()
        self.Mu_Min_Max_X={}
        self.Mu_Min_Max_Y={}

    def Normalize_X(self):
        n_features=self.X.shape[1]
        for i in range(n_features):
            feature=self.X[:,i]
            Mu=np.mean(feature)
            Min=np.min(feature)
            Max=np.max(feature)
            feature=(feature-Mu)/(Max-Min)
            self.Mu_Min_Max_X[i]=np.array([Mu,Min,Max])
            self.X[:,i]=feature      
        return self.X.copy()

    def Normalize_Y(self):
        
        self.Y.reshape(self.Y.shape[0],self.Y.ndim)
        n_targets = self.Y.shape[1]
        for i in range(n_targets):
            target=self.Y[:,i]
            Mu=np.mean(target)
            Min=np.min(target)
            Max=np.max(target)
            target=(target - Mu)/(Max-Min)
            self.Mu_Min_Max_Y[i]=np.array([Mu,Min,Max])
            self.Y[:,i]=target
        return self.Y.copy()
    
    def get_X_trans_parms(self):
        return self.Mu_Min_Max_X
    def get_Y_trans_parms(self):
        return self.Mu_Min_Max_Y