from utils import Star
from utils import *

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


R     = []
T     = []
M     = []
L     = []

test_L_1 = np.array([4.226e4,5.522e3,5.353e2,1.756e1])*L_sun
test_T_1 = np.array([3.478e4,2.513e4,1.728e4,9.505e3])
'''
https://link.springer.com/content/pdf/10.1007%2FBF00642595.pdf
'''


test_L_2 = 10**(np.array([2.92 ,2.99 ,2.44  ,2.48]))*L_sun
test_T_2 = np.array([1.6e4,1.5e4,1.15e4,1.4e4])
'''
https://link.springer.com/content/pdf/10.1007%2FBF00648813.pdf
'''

temp_range = np.linspace(0.3e7,3.e7,10)
#temp_range = [2.45e7]
count = 0
for temp in temp_range:
    output = Star().MS_star(temp,save=True,file_name='./output/{}_{}E6_Kelvin.csv'.format(count,int(temp*1e-6)))
    L.append(output[0])
    M.append(output[1])
    T.append(output[2])       # initial surface temperature estimate
    R.append(output[3])
    count += 1

R = np.array(R)
T = np.array(T)
L = np.array(L)
M = np.array(M)

print('Radius;',R)

T_f = (L/(4*pi*R*R*sigma))**(1/4.)
 
#print('Surface Temperature', T_f)
 
plt.figure(1)
#plt.loglog(test_T_1, test_L_1,'ro')
#plt.loglog(test_T_2, test_L_2,'go')
#plt.loglog(T_f,L, 'ko')
plt.loglog(T_f,L,'yo')
plt.gca().invert_xaxis()
plt.xlabel('Temperature [K]')
plt.ylabel('Luminosity [W]')
plt.show()


output_list = np.concatenate((R.reshape([-1,1]),L.reshape([-1,1]),M.reshape([-1,1]),T.reshape([-1,1])),axis=1)

np.savetxt('Main_Sequence_HIRES.txt', output_list, delimiter=',', header='R, L, M, T')

print("Finished everything....")

            


