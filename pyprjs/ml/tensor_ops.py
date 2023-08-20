import numpy as np
import tensorflow as tf

x = tf.zeros(shape=(2, 1))
print(x)

y = tf.random.normal(shape = (3, 1), mean =0, stddev=1)
print('\n')
print(y)

# variable tensors

z = tf.Variable(initial_value=tf.random.normal(shape = (3,1)))

print(z)
print('Assigned var: \n')

z.assign(tf.ones((3,1))) 

print(z)

# assign_add & assin_sub basically add and substract from tensors (like -= or +=)

