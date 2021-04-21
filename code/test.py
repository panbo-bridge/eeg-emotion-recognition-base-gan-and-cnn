# import numpy as np
# a = [[1,2,3,3,33,3,3,33],[4,5,6,4,4,5,6,9]]
# a = np.array(a)
# a = a.reshape(2,4,-1)
#
# print(a.shape)
# print(a)
# a = a.reshape(2,2,4)
# print(a)
a = [1,2,3]
a = a[-1]
print(a)

# from __future__ import print_function
# import numpy as np
# from sklearn.preprocessing import normalize
#
#
# x = np.array([1, 2, 3, 4], dtype='float32').reshape(1,-1)
#
# print("Before normalization: ", x)
#
# options = ['l1', 'l2', 'max']
# for opt in options:
#     norm_x = normalize(x, norm=opt)
#     print("After %s normalization: " % opt.capitalize(), norm_x)