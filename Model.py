class Model():
    __layers:list[Layer] = None
    __A:list[np.array] = None
    __Z:list[np.array] = None
    __Y:list[np.array] = None 
    __X = None
    cost_array = []
    iters = []
    m:int = None
    
    def __init__(self, *layers:Layer, X,Y,alpha=0.01):
        self.__layers = list(layers)
        self.__Y = Y
        self.__X = X
        self.alpha = 1
        self.m = X.shape[1]
        
    def fit(self):
        self.__layers[0].set_num_of_previous(self.__X.shape[0])
        self.__Z = [None for x in range(len(self.__layers))] 
        self.__A = [None for x in range(len(self.__layers)+1)] 
        for i in range(1,len(self.__layers)):
            self.__layers[i].set_num_of_previous(self.__layers[i-1].get_num_of_neurons())
        self.__A[0] = self.__X
    
    def __forward_propagate(self,X=0,test = False):
        if test:
            self.__A[0] = X
        for i in range(len(self.__layers)):
            self.__Z[i] = np.dot(self.__layers[i].get_weights(),self.__A[i]) + self.__layers[i].get_bias()
            self.__A[i+1] = np.tanh(self.__Z[i]) if i != len(self.__layers)-1 else sigmoid(self.__Z[i])
            
    def __back_propagate(self):
        dA = (-self.__Y/self.__A[len(self.__A)-1]) + ((1-self.__Y)/(1-self.__A[len(self.__A)-1]))
        for i in range(len(self.__layers)-1,0,-1):
            if i == len(self.__layers)-1:
                dZ = dA*(sigmoid(self.__Z[i])*(1-sigmoid(self.__Z[i])))
            else:
                dZ = dA*(1-(np.tanh(self.__Z[i])**2))
            
            dW = (1/self.m) * np.dot(dZ,self.__A[i].T)
            db = (1/self.m) * np.sum(dZ, axis=1,keepdims=True)
            dA = np.dot(self.__layers[i].get_weights().T,dZ)            
            self.__layers[i].set_weights(self.__layers[i].get_weights()-self.alpha*dW)
            self.__layers[i].set_bias(self.__layers[i].get_bias()-self.alpha*db)
            
    def compute_cost(self):
        return (-1 / self.m) * (np.sum(self.__Y*np.log(self.__A[len(self.__A)-1])+ (1-self.__Y)*np.log(1-self.__A[len(self.__A)-1])))
    
    def train(self, num_of_iterations):
        for i in range(num_of_iterations):
            self.__forward_propagate()
            self.__back_propagate()
            if i % 1000 == 0:
                cost = self.compute_cost()
                print(f"The cost is = {cost}")
                self.cost_array.append(cost)
                self.iters.append(i)
    
    def graph_cost(self):
        sns.scatterplot(y= self.cost_array,x=self.iters)
        plt.show()
    
    def test(self,X_test,Y_test):
        self.__forward_propagate(X_test,True)
        np.putmask(self.__A[len(self.__A)-1],self.__A[len(self.__A)-1]>0.5,1)
        np.putmask(self.__A[len(self.__A)-1],self.__A[len(self.__A)-1]<=0.5,0)
        return float((np.dot(Y_test, self.__A[len(self.__A)-1].T) + np.dot(1 - Y_test, 1 - self.__A[len(self.__A)-1].T)) / float(Y_test.size) * 100)   
