from scipy.stats import qmc
import numpy as np

# define the array
array = np.random.rand(2_000_000, 500)

# generate sobol sequence
engine = qmc.Sobol(d=1, scramble=False)
sample = engine.random_base2(m=23)  # 2^m points (m=23 is 8,388,608 points)

# Scale the Sobol sequence to the range of the array indices
indices = (sample * len(array)).astype(int).ravel()

if __name__ == '__main__':
    for i in indices[:3]:
        rand_index = (sample[i] * len(array)).astype(int)[0]
        print(rand_index)
    print('len(indices):', len(indices))
    print('len(array):', len(array))
    print('len(set(indices)):', len(set(indices)))
