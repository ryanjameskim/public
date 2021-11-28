
import numpy as np
import math

class multilinearexpansion:
    '''Takes a list of outputs of a function f mapping k dimension {0,1} and a field_n
    and computes the multilinear expansion of the function up to that field_n  

    f_outputs: list of integers, size of 2^k
    field_n: any modular n

    '''

    def __init__(self, f_outputs, field_n):
        self.outputs = np.array(f_outputs)
        self.field_n = field_n
        
        #correct outputs by field n
        self.outputs = self.outputs % field_n

        #check correct number of outputs
        if len(self.outputs) % 2 != 0:
            print(f'Number of outputs is {len(self.outputs)} not power of 2')
        self.dimension = int(np.ceil(np.log2(len(self.outputs))))

        #build necessary domain vectors
        self.domain = self.build_domain(self.dimension, 2)
        self.mle_domain = self.build_domain(self.dimension, field_n)

        #use Lemma 3.7 to build out mle table
        mle_table = []
        for r_i in self.mle_domain:
            mle_table.append(self.calculate_f_r(r_i))

        #print original function table and then mle
        print('*********ORIGINAL FUNCTION***************')
        self.print_outputs(2, self.outputs)
        print('*********MLE FUNCTION********************')
        self.print_outputs(self.field_n, mle_table)

        
    def build_domain(self, n_dims, space):
        result = []
        domain = [np.array([i]) for i in range(space)]   #build a one dimensional space
        for i in range(n_dims-1):    #build zeros for each new dimension
            temp_domain = []     #create an empty list
            for j in range(space):   #for each unit in the space
                for k in range(len(domain)):                        #for each item in the existing domain
                    temp = np.expand_dims(domain[k], axis=0)        #expand the item in the front
                    temp_domain.append(np.append(j,temp))           #and add the existing j value to the front, append to temp_domain
            domain = temp_domain #replace domain for each dimension
        return domain

    def calculate_f_r(self, r):         #each output for new domain r requires a pass of each of the original outputs
        output_chi_sum = 0
        for f_output, w_input in zip(self.outputs, self.domain):
            output_chi_sum += f_output * self.chi(r, w_input)
        return output_chi_sum % self.field_n

    def chi(self, r, w):
        #r and w are each vectors of n-dimensions
        chi = r*w + (1-r) * (1-w)
        return np.product(chi)
    
    def print_outputs(self, space, outputs):
        #check if 2d
        if self.dimension != 2:
            print('Unable to print multiple dimensions')
            return
        #print table if 2d
        print('   |', end='')
        for i in range(space):
            print(f'   {i} |', end='')
        else:
            print('\n')
        for row in range(len(outputs) // space):
            print(f' {row} |', end='')
            for column in range(space):
                print(f'   {outputs[row*space + column]} |', end='')
            else:
                print('\n')

    def mle_equation(self):
        '''can only handle hardcoded coefficient matrixes now
        trying to figure out a dynamic solution
        '''

        if self.dimension == 2:
            #hardcode
            self.coff_matrix = np.array(
                                       [[1, -1, -1, 1],     #(0, 0)
                                        [0,  0,  1,-1],     #(0, 1)
                                        [0,  1,  0,-1],     #(1, 0)
                                        [0,  0,  0, 1]])    #(1, 1)

        if self.dimension == 3:
            #hardcode
            self.coff_matrix = np.array(
                                       [[1, -1, -1, -1,  1,  1,  1, -1],
                                        [0,  0,  0,  1,  0, -1, -1,  1],
                                        [0,  0,  1,  0, -1,  0, -1,  1],
                                        [0,  1,  0,  0, -1, -1,  0,  1],
                                        [0,  0,  0,  0,  0,  0,  1, -1],
                                        [0,  0,  0,  0,  0,  1,  0, -1],
                                        [0,  0,  0,  0,  1,  0,  0, -1],
                                        [0,  0,  0,  0,  0,  0,  0,  1]])            


        #transpose the list of outputs into column matrix -- this is bracketed to add another dimension
        self.trans_outputs = np.transpose([self.outputs])
        #element-wise multiply the coff matrix and the transposed outputs, then sum by column
        self.multilinearequation = np.sum(np.multiply(self.coff_matrix, self.trans_outputs), axis=0)

        print(self.multilinearequation)




def main():
    mle = multilinearexpansion([1,2,1,4],5)
    mle.mle_equation()
    mle2 = multilinearexpansion([3,4,1,2],5)
    mle2.mle_equation()
    mle3 = multilinearexpansion([1,5,2,6,3,7,4,8],7)
    mle3.mle_equation()

if __name__ == "__main__":
    main()


