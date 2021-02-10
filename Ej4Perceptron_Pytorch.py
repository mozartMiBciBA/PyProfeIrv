#Fuente: https://www.youtube.com/watch?v=lkSZ1PTAvUY

import numpy as np
import torch

print("np.__version__")
print(np.__version__)

print("torch.__version__")
print(torch.__version__)


# Tensor 2 dimensiones de 4 x 2 elementos reyeno de unos
y = torch.ones(4,2)

r= torch.ones(3,3)

print("y")
print(y)

print("r")
print(r)

# operación entre tensores

z= y+y+y

print("z")
print("La suma es una operación elemento por elemento")
print(z)


#acceder a los elementos de los tensores se hace igual como los arrays de numpy
print("z[1][1]")
print(z[1][1])

print(z[:,1:])


#Función reshape para adecuar los tensores a las arquitecuras de kas redes
#Es para aplanar una matriz (imagen)
# O de un vector aplanado regresarlo a la matriz que representa una imagen.

print("z.size()")
print(z.size())


#Adecuar z a una matriz de 3x3 elementos
#Si falta un elemento pytorch los rellena con 0
print("z.resize_(3,3)")
# resize_  Modifica el tensor z
print(z.resize_(3,3))


################## Pasar de tensores a numpy y viceversa  ###########3######


# Generar un array en np
a = np.random.rand(4,3)
print("a")
print(a)

# Convertir a tescor con pytorch
A=torch.from_numpy(a)
print("A")
print(A)

# Covertir a np
b= A.numpy()
print("b")
print(b)

"""
Advertencia! Ten cuidado por que resulta que las variables convertidas de pytorch a numpy comparte memoria asi
que si modificas uno se modificará el otro!
"""

print("A")
print(A)

#Multiplicamos el tensor A
A.mul_(2)
print("A")
print(A)

print("b")
print(b)


########### Implementación de nuestro primer Perceptron#######
#Entradas x en np 1 dimensión con 3 elementos
x_np= np.array([2.0, 3.0, 4.0])
w_np= np.array([0.1, 0.1, 0.2])
# El sesgo
b_np= np.array([1.0])

# se crea h_np en numpy
h_np = None
print(h_np)


# Convierte los arrays de np a tensores de pytorch para hacer los calculos
X = torch.from_numpy(x_np)
print("X")
print(X)
W = torch.from_numpy(w_np)
print("W")
print(W)
B = torch.from_numpy(b_np)
print("B")
print(B)


# realiza las operaciones necesarias y calcula la combinación lineal h= w . x + b
# algunas funciones útiles son torch.add() y torch.dot()
# Primero se realiza el producto punto entre X y W con torch.dot
# Despues se realiza la suma del resultado del producto punto mas B con pytorch.add

H =  torch.add(torch.dot(W,X),B)
print("H")
print(H)


# Pasa h por la función de activación segmoide torch.sigmoid()
#Nuestro perceptron dadas las entradas anteriores X, W y B es Y
Y = torch.sigmoid(H)
print("Y")
print(Y)


# Regresar Y a numpy

y_np= Y.numpy()
print("y_np")
print(y_np)
