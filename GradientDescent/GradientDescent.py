from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import time

X = [i for i in range(-100,100)]
Y = [i for i in range(-100,100)]

x = np.asarray(X)
y = np.asarray(Y)

X, Y = np.meshgrid(x,y)

print(X)
print(Y)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
Z = X*X + Y*Y + 6
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=30, cstride=30)
x = 98
y = 98
z = x*x+y*y
# plt.plot(x,y,z,'ro')
z_ = []
x_ = []
y_ = []
z_.append(z)
x_.append(x)
y_.append(y)
count = 0
count_t = 0
while True:
# calculate derivates wrt to first point
    count_t += 1
    dZdx = 2*x #slope in x direction
    dZdy = 2*y #slope in y direction

    lr = 0.1

    slope_updated = 2*x

    slope_updated_with_lr = lr*2*x
    slope_updated_with_lr_y = lr*2*y

    x_up = x-slope_updated_with_lr
    y_up = y-slope_updated_with_lr_y
    z_up = x_up*x_up+y_up*y_up + 6
    
    if z_up >= z:
        count+=1

    z = z_up
    y = y_up
    x = x_up

    z_.append(z)
    x_.append(x)
    y_.append(y)

    if count == 3:
        break

print("Total Count is {}".format(count_t))
print("Final Cost is {}".format(z_[-1]))
plt.plot(x_,y_,z_,'r*')
plt.show()

# Now start the gradient descent algorithm


