import os
import numpy as np

class nodemap:
    
    def __init__(self, nodemap_path=None, nodevector_path=None):

        if nodemap_path is not None:
            assert os.path.isfile(nodemap_path)
            with open(nodemap_path, 'r') as f:

                # read lines, skipping header
                lines = f.readlines()[1:]
                for cnt, line in enumerate(lines):
                    lines[cnt] = line.split('\t')[1]
                self.map = tuple(lines)

        if nodevector_path is not None:
            assert os.path.isfile(nodevector_path)
            with open(nodevector_path, 'r') as f:
                # read in vectors to list
                vec_list = f.readlines()[1:]
                vec_list = tuple(x.split() for x in vec_list)
                
                immut_vec_list = list()
                for entry in vec_list:
                    mytuple = tuple(float(x) for x in entry)
                    immut_vec_list.append(mytuple)
                immut_vec_list = sorted(immut_vec_list)
                # generate immutable vector
                dt = np.dtype('float')
                self.n_vectors = tuple(x[1:] for x in immut_vec_list)
                self.numpy_vectors = tuple(np.array(x, dtype=dt) for x in self.n_vectors)
    
    def __getitem__(self, index):
        assert self.map
        return self.map[index]



if __name__ == "__main__":
    filename = '../examples/test.edgemap'

    nodemap_test = nodemap(filename)

    print(nodemap_test[5])
