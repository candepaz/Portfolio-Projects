import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(t,x1,x2,x3):
    '''
    Function: f(t,x)
    Purpose: sets up the ODE-2D x'=f(t,x)
    Arg-in: t(float)
            x (list)
    Arg-out: f(list)
    '''
    return [l * x1 * x2/N - u * x1, -l * x1 * x2/N, u * x1]

## Initial conditions
t0 = 0.0
x10 = 10.0
x20 = 990.0
x30 = 0.0

## Constants
l = 1.0
u = 0.3
N = 1000

h = 0.1
tf = 30
n = int((tf-t0)/h)
t = np.zeros(n+1)
x = np.zeros((n+1, 3))  #matriz de n+1 x 3 llena de ceros


t[0] = t0
x[0] = [x10, x20, x30]

value_found = False

for i in range(n):
    t[i+1] = t[i] + h
    p = np.multiply(f(t[i], x[i,0], x[i,1], x[i,2]),h)
    k1 = p[0]
    l1 = p[1]
    j1 = p[2]
    q = np.multiply(f(t[i+1], x[i,0] + k1, x[i,1] + l1, x[i,2] + j1),h)
    k2 = q[0]
    l2 = q[1]
    j2 = q[2]
    x[i+1,0] = x[i,0] + (k1 + k2)/2
    x[i+1,1] = x[i,1] + (l1 + l2)/2
    x[i+1,2] = x[i,2] + (j1 + j2)/2
    

# =============================================================================
# # Table print-out
# for i in range(n+1):
#     print(i,'%f.'%t[i],'%f.'%x[i,0],'%f.'%x[i,1],'%f.'%x[i,2])
# =============================================================================
    
    
'''
x1_max = x[i,0] # Saves first value of i(t)
x2_ant = x[i,1] # Saves first value of s(t)
x3_ant = x[i,2] # Saves first value of r(t)

for i in range(1,500):
    # Infected max. Gets time when i(t) is a MAX.
    if (x[i,0] > x1_max):
        tpico = t[i]
    else:
        x1_ = 
 
    else:
        
    
    # Obtener tiempo aprox donde i(t) < s(t)
    if (x[i,0] > vmax [2]):
        t_i_con_s = vmax[0]
    else:
        vmax[2] = x[i,1]
'''     

#print(tpico) # imprime valor de t dónde se produce el pico de i(t)
#print(t_i_con_s) # imprime valor de t dónde se produce i(t) intersección (aprox) con s(t) (la última de las veces que se cortan)
'''
line_up, = plt.plot(t,x[:,:1], label='función')
line_down, = plt.plot(t,x[:,1:2], label='derivada')
plt.legend(handles=[line_up, line_down])
'''
plt.plot(t,x[:,:1], label='i(t)')
plt.plot(t,x[:,1:2], label='s(t)')
plt.plot(t,x[:,2:3], label='r(t)')
plt.legend()
plt.xlabel('$days$')
plt.ylabel('$cases$')
#plt.savefig('grafico.jpg')
plt.show()

df = pd.DataFrame(x, index = t)
df.reset_index(inplace = True)
df.head()

cols = ['t', 'i', 's', 'r']

df.to_csv(r"C:/Users/Cande/Desktop/Cande/portfolio/SIRx_Model_data.csv", header = cols, index = False, sep=',')




