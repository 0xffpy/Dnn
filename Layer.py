class Layer():
    __weights :np.array = None
    __bais: np.array = None
    __num_of_nuerons:int = None
    __num_of_previous:int = None
    
    def __init__(self, num_of_neurons ,num_of_previous = 0):
        np.random.seed(19)
        self.__weights = None if num_of_previous == 0 else np.random.randn(num_of_neurons,num_of_previous)*0.01
        self.__bais = np.zeros(shape = (num_of_neurons,1))
        self.__num_of_nuerons = num_of_neurons
        self.__num_of_previous = num_of_previous
        
    def set_num_of_previous(self, num):
        self.__num_of_previous = num
        self.set_weights(np.random.randn(self.__num_of_nuerons,self.__num_of_previous)*0.01)
        
    def get_num_of_previous(self):
        return self.__num_of_previous
    
    def get_num_of_neurons(self):
        return self.__num_of_nuerons
    
    def get_weights(self):
        return self.__weights
    
    def set_weights(self, weight):
        self.__weights = weight
    
    def get_bias(self):
        return self.__bais
    
    def set_bias(self,bais):
        self.__bais = bais
