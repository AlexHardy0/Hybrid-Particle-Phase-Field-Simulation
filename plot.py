import numpy as np
import matplotlib.pyplot as plt

#parameters copied from hybrid particle code
dx = 0.5
dt = 0.001
		
Nt = 2000000
nt_save = 1000

Nx, Ny = 64, 256  # number of lattice points should be power of two
Lx, Ly = Nx*dx, Ny*dx
print(f'Lx = {Lx}; Ly = {Ly}')

C = 5.0
G = -0.05
B = 500.0
Np = 8192 

# create a 1D and 2D array of x- and y- coordinates (useful for plotting)
x = np.arange(0, Nx)*dx
y = np.arange(0, Ny)*dx
Y, X = np.meshgrid(y, x)

nt = 1500000
fig, ax = plt.subplots(1, 2, figsize=(7,8))

# SETTING UP PLOTTING 

# set labels
for n in range(0, 2):
	ax[n].set_xlabel('$x$', fontsize=16)
	ax[n].set_ylabel('$y$', fontsize=16)
	ax[n].set_xlim([-0.5*dx, Lx-0.5*dx])
	ax[n].set_ylim([-0.5*dx, Ly-0.5*dx])
	ax[n].set_xticks(np.arange(0, Lx, 10))
	ax[n].set_yticks(np.arange(0, Ly, 10))
	ax[n].tick_params(axis='both', which='major', labelsize=12)
	ax[n].set_aspect(1)

ax[0].set_title('$\{\mathbf{r}_i\}$', fontsize=16)
ax[1].set_title('$\mathbf{u}(\mathbf{r})$', fontsize=16)

phi = np.loadtxt(f'./data/phi{nt}.txt')
graph00 = ax[0].pcolormesh(X, Y, phi, vmin=-1.2, vmax=1.2, shading='auto', cmap='summer')
#fig.colorbar(graph00, ax=ax[0], shrink=0.8)

graph10 = ax[1].pcolormesh(X, Y, phi, vmin=-1.2, vmax=1.2, shading='auto', cmap='summer')
cbar = fig.colorbar(graph10, ax=ax[1], shrink=0.8)
cbar.ax.tick_params(labelsize=12)

px = np.loadtxt(f'./data/px{nt}.txt')
py = np.loadtxt(f'./data/py{nt}.txt')
graph01, = ax[0].plot(px, py, '.', color='black', markersize=2)

ux = np.loadtxt(f'./data/ux{nt}.txt')
uy = np.loadtxt(f'./data/uy{nt}.txt')
graph11 = ax[1].quiver(X[::2,::2], Y[::2,::2], ux[::2,:Ny:2], uy[::2,:Ny:2], angles='xy', scale_units='xy', scale=0.02)

plt.savefig('snapshot.png')



# Comparison with theory

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
bins = np.arange(0, Ny+1)*dx - 0.5*dx

h = Ly/2.0 - 0.49
nt_final = Nt-nt_save
		
print(nt_final)
		
phi_data = np.loadtxt(f'./data/phi{nt_final}.txt')
count = 0.0
py_data = []
for nt in np.arange(int(Nt/2), Nt, nt_save):
	py_data = np.concatenate((py_data, np.loadtxt(f'./data/py{nt_final}.txt')))
	count += 1.0
print(count)
print(np.shape(py_data))

# tanh expression
tanh_expr = np.tanh(h - y)

# calculate phi 
trial_psi = np.exp(-G*y + C*tanh_expr)
c1 = (1/np.sum(dx * trial_psi)) * (Np/Nx)
	
split = int(h/dx)
exp1 = np.exp((-G-2)*h - C - 2*y[0:split])
exp2 = np.exp((-G-2)*h - C + 2*y[0:split])
exp3 = np.exp((-G-2)*h + C - 2*y[0:split])
exp4 = np.exp((-G-2)*h + C + 2*y[0:split])
exp5 = np.exp(-2*y[0:split] + C)
exp6 = np.exp(-G*y[0:split] + C)

exp7 = np.exp((-G+2)*h - C - 2*y[split:])
exp8 = np.exp((-G+2)*h + C - 2*y[split:])
exp9 = np.exp((-G-2)*h - C - 2*y[split:])
exp10 = np.exp((-G-2)*h + C - 2*y[split:])
exp11 = np.exp(-2*y[split:] + C)
exp12 = np.exp(-G*y[split:] - C)

G1 = -G + 2
G2 = G - 2
G3 = G + 2
G4 = -G -2
fraction = c1/((2*G**2)-8)

phi1_left = -((G1 * exp1) + (G1 * exp2) + (G2 * exp3) + (G2*exp4) - (2*G*exp5) + (4*exp6))*fraction
phi1_right = ((G2 * exp9) + (G3 * exp7) + (G1 * exp10) + (G4*exp8) + (2*G*exp11) - (4*exp12))*fraction

phi1 = np.ones_like(y)
phi1[0:split] = phi1_left
phi1[split:] = phi1_right

phi_calc = tanh_expr + (C/B)*phi1

psi_calc = np.exp(-G*y + C*phi_calc)
c1 = (1/np.sum(dx * psi_calc)) * (Np/Nx)
psi_calc = psi_calc * c1

# plotting
# set labels
for n in range(0, 2):
	ax[n].set_xlabel('$y$', fontsize=16)
	ax[n].set_xlim([-0.5*dx, Ly-0.5*dx])
	ax[n].set_xticks(np.arange(0, Ly, 10))
	ax[n].tick_params(axis='both', which='major', labelsize=12)

ax[0].set_ylabel("$\\phi$", fontsize=16)
ax[1].set_ylabel("$\\psi$", fontsize=16)
ax[0].set_ylim([-1.2, 1.2])

ax[0].plot(y ,phi_data[0,:], linewidth=2.0, label='hybrid particle-phase field simulation')
ax[0].plot(y, phi_calc, linewidth=2.0, label='perturbative solution from theory')
ax[0].plot(y, y*0, color='black', linewidth=0.5)
ax[1].hist(py_data, bins=bins, weights=np.ones(np.shape(py_data))/count, label='hybrid particle-phase field simulation')
ax[1].plot(y, psi_calc*Lx, linewidth=2.0, label='perturbative solution from theory')

ax[0].legend(loc='lower left', fontsize=12)
ax[1].legend(loc='upper right', fontsize=12)

plt.savefig('comparison.png')


fig, ax = plt.subplots(figsize=(8, 4))

ax.set_xlim([-0.5*dx, Ly-0.5*dx])
ax.set_ylim([-0.04, 0.04])
ax.set_xlabel('$y$', fontsize=24)
ax.set_ylabel('$\phi_{theory}-\phi_{simulation}$', fontsize=24)
ax.set_xticks(np.arange(0, Ly, 20))
ax.set_yticks(np.arange(-0.04, 0.0401, 0.02))
ax.tick_params(axis='both', which='major', labelsize=24)

ax.plot(y, phi_calc-phi_data[0,:], linewidth=3.0)
plt.savefig('diff.png')
