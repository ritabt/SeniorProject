import numpy as np

# random sampling for learning from experience replay
class Exp():
    def __init__(self, obs_size, max_size, img_size=None):
        self.obs_size = obs_size
        self.img_size = img_size
        self.num_obs = 0
        self.max_size = max_size
        self.mem_full = False
        
        if img_size is None:
            # memory structure that stores samples from observations
            self.mem = {'s'        : np.zeros((self.max_size, self.obs_size), dtype=np.float32),
                        'a'        : np.zeros((self.max_size, 1), dtype=np.int32),
                        'r'        : np.zeros((self.max_size, 1)),
                        'done'     : np.zeros((self.max_size, 1), dtype=np.int32)}
        else:
            self.mem = {'s'        : np.zeros((self.max_size, self.img_size[0], self.img_size[1], self.img_size[2]), dtype=np.float32),
                        'a'        : np.zeros((self.max_size, 1), dtype=np.int32),
                        'r'        : np.zeros((self.max_size, 1)),
                        'done'     : np.zeros((self.max_size, 1), dtype=np.int32)}

    # stores sample obervation at each time step in experience memory
    def store(self, s, a, r, done):
        i = self.num_obs % self.max_size
        
        # TODO: test if can remove the : self.mem['s'][i] = s
        self.mem['s'][i] = s
        self.mem['a'][i,:] = a
        self.mem['r'][i,:] = r
        self.mem['done'][i,:] = done
        
        self.num_obs += 1
        
        if self.num_obs == self.max_size:
            self.num_obs = 0 # reset number of observation
            self.mem_full = True

    # returns a minibatch of experience
    def minibatch(self, minibatch_size):
    	# TODO: mem full var seems useless. Check rest of code if used
    	# mem full can be removed 
    	# change line 33 to modulo self.num_obs %= self.max_size
    	# change below to 
        if self.mem_full == False:
            max_i = min(self.num_obs, self.max_size) - 1
        else:
            max_i = self.max_size - 1
        
        # randomly sample a minibatch of indexes
        sampled_i = np.random.randint(max_i, size=minibatch_size)      
                
        if self.img_size is None:
            s      = self.mem['s'][sampled_i,:].reshape(minibatch_size, self.obs_size)
            s_next = self.mem['s'][sampled_i + 1,:].reshape(minibatch_size, self.obs_size)
        else:
            s      = self.mem['s'][sampled_i,:].reshape(minibatch_size, self.img_size[0], self.img_size[1], self.img_size[2])
            s_next = self.mem['s'][sampled_i + 1,:].reshape(minibatch_size, self.img_size[0], self.img_size[1], self.img_size[2])
        a      = self.mem['a'][sampled_i].reshape(minibatch_size)
        r      = self.mem['r'][sampled_i].reshape((minibatch_size,1))
        done   = self.mem['done'][sampled_i].reshape((minibatch_size,1))
        
        return (s, a, r, s_next, done)