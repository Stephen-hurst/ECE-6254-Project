class GameHistory():
    '''Maintain a history of state-action-value tuples, replacing randomly
    when we exceed the max storage'''
    def __init__(self, features, max_size=50000):
        self.max_size = max_size
        self._lock = threading.Lock()
        self._storage = np.zeros((0,features))
        
    def add(self, samples):
        '''Add N rows (samples is a matrix with N rows and some features)'''
        self._lock.acquire()
        try:
            if(self._storage.shape[0] < self.max_size):
                self._storage = np.concatenate([self._storage, samples])
                # chop if we went over
                if(self._storage.shape[0] > self.max_size):
                    self._storage = self._storage[0:self.max_size,:]
            else:
                rows_to_replace = np.random.choice(self.max_size, samples.shape[0])
                self._storage[rows_to_replace,:] = samples
        except e:
            raise e
        finally:
            self._lock.release()
        
    def sample(self, N):
        '''Return N random rows'''
        rows = np.random.choice(self._storage.shape[0], N)
        return self._storage[rows,:]
    
