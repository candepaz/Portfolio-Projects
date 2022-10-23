import numpy as np
import matplotlib.pyplot as plt

def f(t,x1,x2,x3):
    '''
    Function: f(t,x)
    Purpose: sets up the ODE-2D x'=f(t,x)
    Arg-in: t(float)
            x (list)
    Arg-out: f(list)
    '''
    return [l * x1 * x2/N - u * x1, -l * x1 * x2/N, u * x1]

##Lambda
l0 = 0.3
lf = 3.3
s = 0.05
m = np.int((lf-l0)/s)
l_matriz = np.zeros(m+1)
l = l0

l_matriz[0] = l0

##Initial conditions
t0 = 0.0
x10 = 10.0
x20 = 990.0
x30 = 0.0

##Constants
u = 0.3
N = 1000


h = 0.1
tf = 30
n = int((tf-t0)/h)
t = np.zeros(n+1)
x = np.zeros((n+1, 3))  #matriz de n+1 x 3 llena de ceros


t[0] = t0
x[0] = [x10, x20, x30]


t_pico = 0
matriz_duracion_epid = np.zeros(m+1)

for j in range(m):
    l = l_matriz[j]
    print("iterating with lambda = ", l_matriz[j])    
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
        
        
        #For I(t) Max
        if (x[i-1,0] < x[i,0] and x[i,0] > x[i+1,0]):
            t_pico = t[i] 
            matriz_duracion_epid[j] = t_pico*2
            print("max time = ", t_pico)
        
        #For t where I(t) = S(t) 
        if (x[i-1,0] > x[i-1,1] and x[i+1,0] < x[i+1,1]):
            print("2nd time when I(t) = S(t) = ", t[i])
            
        #For t where R(t) = I(t)
        if (x[i-1,0] > x[i-1,2] and x[i+1,0] < x[i+1,2]):
            print("time when I(t) = R(t) = ", t[i]) 
            
        #For t where S(t) = R(t)
        if (x[i-1,1] > x[i-1,2] and x[i+1,1] < x[i+1,2]):
            print("time when S(t) = R(t) = ", t[i])

    l_matriz[j+1] = l_matriz[j] + s  


    plt.plot(t,x[:,:1], label='i(t)')
    plt.plot(t,x[:,1:2], label='s(t)')
    plt.plot(t,x[:,2:3], label='r(t)')
    plt.legend()
    plt.xlabel('$days$')
    plt.ylabel('$cases$')
    plt.savefig('grafico.jpg')
    plt.show()

#print(x)

matriz_duracion_epid[m] = t_pico*2
plt.plot(l_matriz,matriz_duracion_epid)


#for i in range(m+1):
#   print('%f.'% l_matriz[i], '%f.' % matriz_duracion_epid[i])
    
'''
line_up, = plt.plot(t,x[:,:1], label='función')
line_down, = plt.plot(t,x[:,1:2], label='derivada')
plt.legend(handles=[line_up, line_down])
'''

