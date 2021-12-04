
import numpy as np
import math

class multilinearexpansion:
    '''Takes a list of outputs of a function f mapping k dimension {0,1} and a field_n
    and computes the multilinear expansion of the function up to that field_n  

    f_outputs: list of integers, size of 2^k, ordered (0,0,0), (0,0,1), (0,1,0) ... (1,1,0), (1,1,1)
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
        self.mle_table = []
        for r in self.mle_domain:
            self.mle_table.append(self.calculate_f_r(r))

        #use Lemma 3.8 to build out mle table
        self.mle_table2 = []
        for r in self.mle_domain:
            self.mle_table2.append(self.lemma_3_8(r))

        #print original function table and then mle
        print('*********ORIGINAL FUNCTION***************')
        self.print_outputs(2, self.outputs)
        print('*********Lemma 3.7********************')
        self.print_outputs(self.field_n, self.mle_table)
        print('*********Lemma 3.8********************')
        self.print_outputs(self.field_n, self.mle_table2)

        
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

    def lemma_3_8(self, r):
        chi_row = [1]
        for r_i in r[::-1]:  #reversed here to keep order of the final result natural
            #order progression goes:
            #0, 1
            #00, 01, 10, 11
            #000, 001, 010, 011, 100, 101, 110, 111
            chi_row = [(1-r_i) * prev_chi for prev_chi in chi_row] + [(r_i) * prev_chi for prev_chi in chi_row]
        return np.sum(np.multiply(np.array(chi_row), self.outputs)) % self.field_n
    
    def print_outputs(self, space, outputs):
        #check if 2d
        if self.dimension != 2:
            print('Unable to print multiple dimensions')
            return
        #print table if 2d
        print('   |', end='')
        for i in range(space):
            print(f'  #{i} |', end='')
        else:
            print('\n')
        for row in range(len(outputs) // space):
            print(f'#{row} |', end='')
            for column in range(space):
                print(f'   {outputs[row*space + column]} |', end='')
            else:
                print('\n') 
        
    def print_mle(self):
        '''finds a coefficient matrix for a general mle equation

        WARNING: Ordering of terms is tree-like with last dimensions pushing
        front so thus needs outputs reordered in order to work

        Example order of 3 variable cofficients:
        (constant) (x_1) (x_2) (x_2*x_1) (x_3) (x_3*x_1) (x_3*x_2) (x_3*x_2*x_1)

        Function pretty prints at end to avoid confusion
        '''
        #initialize cofficient matrix
        base_matrix = np.array([[1,-1],[0,1]])
        self.coff_matrix = base_matrix.copy()
        
        #initialize ordering helper
        ordering_helper = ['0', '1']
        result_sort = ordering_helper.copy()
        
        #per dimension
        for count in range(self.dimension - 1):
            temp_result = []
            for row in self.coff_matrix:                        #multiply by (1, -1)
                temp_result.append(np.concatenate([row * base_matrix[0][0], row * base_matrix[0][1]]))
            for row in self.coff_matrix:                       # multiply by (0, 1)
                temp_result.append(np.concatenate([row * base_matrix[1][0], row * base_matrix[1][1]]))
            self.coff_matrix = np.array(temp_result)
            result_sort = [order_ele + res_ele for res_ele in result_sort for order_ele in ordering_helper]
            
        #for pretty printing coefficient names
        x_nums = list(range(1,self.dimension+1))
        x_names = []
        for bi_str in result_sort:
            end_str = ''
            for i, ele in enumerate(bi_str):
                if int(ele) == 1:
                    end_str += f'(x_{x_nums[i]})'
            x_names.append(end_str)

        result_sort = [int(res_ele,2) for res_ele in result_sort]    #convert binaries to numbers
        self.reordered_outputs = np.transpose([self.outputs[result_sort]])   #reorder and transpose to column

        #calculate equation
        self.mle_dynamic = np.sum(np.multiply(self.coff_matrix, self.reordered_outputs), axis=0)

        #print Equation
        self.mle_pretty = 'Equation: '
        for i, ele in enumerate(self.mle_dynamic):
            if ele != 0:
                sign = ''
                if ele > 0:
                    sign = '+'
                else:
                    sign = '-'
                if i == 0:
                    sign = ''
                self.mle_pretty += f'{sign} {abs(ele)}{x_names[i]} '
        print(self.mle_pretty)
        return

    
    def mle_equation_old(self):
        '''can only handle hardcoded coefficient matrixes, i've kept for testing
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
                                       [[1, -1, -1, -1,  1,  1,  1, -1], #(0, 0, 0)
                                        [0,  0,  0,  1,  0, -1, -1,  1], #(0, 0, 1)
                                        [0,  0,  1,  0, -1,  0, -1,  1], #(0, 1, 0)
                                        [0,  0,  0,  0,  0,  0,  1, -1], #(0, 1, 1)
                                        [0,  1,  0,  0, -1, -1,  0,  1], #(1, 0, 0)
                                        [0,  0,  0,  0,  0,  1,  0, -1], #(1, 0, 1)
                                        [0,  0,  0,  0,  1,  0,  0, -1], #(1, 1, 0)
                                        [0,  0,  0,  0,  0,  0,  0,  1]])#(1, 1, 1)            


        #transpose the list of outputs into column matrix -- this is bracketed to add another dimension
        self.trans_outputs = np.transpose([self.outputs])
        #element-wise multiply the coff matrix and the transposed outputs, then sum by column
        self.multilinearequation = np.sum(np.multiply(self.coff_matrix, self.trans_outputs), axis=0)
        
        print(f'Hardcoded solution: {self.multilinearequation}')



def main():
    mle = multilinearexpansion([1,2,1,4],5)
    mle.mle_equation_old()
    mle.print_mle()
    mle2 = multilinearexpansion([3,4,1,2],5)
    mle2.mle_equation_old()
    mle2.print_mle()
    mle3 = multilinearexpansion([1,5,2,6,3,7,4,8],11)
    mle3.mle_equation_old()
    mle3.print_mle()

if __name__ == "__main__":
    main()

