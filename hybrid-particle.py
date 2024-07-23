import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import shutil

class HybridParticle():
	
	''' Initialization '''
	def __init__(self):
		# SYSTEM PARAMETERS
		# space parameters
		self.dx = 0.5
		self.dt = 0.001
		
		self.Nx, self.Ny = 64, 256  # number of lattice points should be power of two
		self.Lx, self.Ly = self.Nx*self.dx, self.Ny*self.dx
		print(f'Lx = {self.Lx}; Ly = {self.Ly}')

                # time parameters
		self.Nt = 2000000
		self.nt_save = 1000
		
		# create a 1D and 2D array of x- and y- coordinates (useful for plotting)
		self.x = np.arange(0, self.Nx)*self.dx
		self.y = np.arange(0, self.Ny)*self.dx
		self.Y, self.X = np.meshgrid(self.y, self.x)
		
		# physical parameters
		self.Np = 8192       #no. of particles
		self.M_phi = 0.002   #mobility constant for phi field
		self.B = 500.0       #fluid parameter B
		self.C = 5.0         #confinement parameter C
		self.G = -0.05       #gravity constant G
		self.eta = 10.0      #viscosity constant eta
		
		# initialize particles, phi
		self.px = np.random.uniform(0.0, self.Lx, self.Np)
		self.py = np.random.uniform(0.0, self.Ly/2, self.Np)		
		self.psi = np.zeros((self.Nx, self.Ny), dtype='float64')
		self.phi = np.zeros((self.Nx, self.Ny), dtype='float64')
		self.phi[:,:] = -1.0
		self.phi[:,0:int(self.Ny/2)] = 1.0
		self.mu_phi = np.zeros((self.Nx, self.Ny), dtype='float64')
		
		# initialize fluid velocity u (note that array size is doubled in y-axis)
		self.ux = np.zeros((self.Nx, 2*self.Ny), dtype='float64')
		self.uy = np.zeros((self.Nx, 2*self.Ny), dtype='float64')

		# define a 2D array of wavevectors, note that q is arranged in a peculiar way in Python
		self.qx = (2*np.pi/self.Lx)*np.concatenate((np.arange(0, self.Nx/2, 1), np.arange(-self.Nx/2, 0, 1)))
		self.qy = (np.pi/self.Ly)*np.concatenate((np.arange(0, self.Ny, 1), np.arange(-self.Ny, 0, 1)))
		self.qy, self.qx = np.meshgrid(self.qy, self.qx)
		self.q2 = self.qx*self.qx + self.qy*self.qy
		self.q4 = self.q2*self.q2
				
	''' Calculate the derivatives of q (where q can be phi or any field) '''
	def calculate_derivative(self, q, type, negative):
                #negative is used to flip the sign of the wall values if needed for conservation
		if negative:
			factor = -1
		else:
			factor = 1

		# make padded array to add extra layers on each 4 sides of q - ghost walls
		Q = np.zeros((self.Nx+2, self.Ny+2))
		Q[1:-1,1:-1] = q
		Q[0,1:-1] = q[-1,:]      # q(i=-1) = q(i=Nx-1)  periodic boundary condition at x=0 and x=Lx
		Q[-1,1:-1] = q[0,:]      # q(i=Nx) = q(i=0)
		Q[:,0] = Q[:,1]*factor   # q(j=-1) = q(j=0)  wall boundary condition at y=0 and y=Ly
		Q[:,-1] = Q[:,-2]*factor # q(j=Ny) = q(y=Ny-1)  give negative vector for J

		if type == 'wrt_x':
                        # first derivative in x direction
			dqdx = (Q[2:,1:-1] - Q[0:-2,1:-1])/(2*self.dx)
			return dqdx
		
		if type == 'wrt_y':
                        # first derivative in y direction
			dqdy = (Q[1:-1,2:] - Q[1:-1,0:-2])/(2*self.dx)
			return dqdy
			
		if type == 'laplacian':
                        # laplacian
			dqdx2 = (Q[2:,1:-1] - 2*Q[1:-1,1:-1] + Q[0:-2,1:-1])/(self.dx*self.dx)
			dqdy2 = (Q[1:-1,2:] - 2*Q[1:-1,1:-1] + Q[1:-1,0:-2])/(self.dx*self.dx)
			return dqdx2 + dqdy2
		
	''' Update phi '''
	def update_phi(self):
                
		# calculate particle density psi by histogramming all positions
		edges_x = np.arange(0, self.Nx+1) - 0.5  # define bin edges
		edges_y = np.arange(0, self.Ny+1) - 0.5
		self.psi = np.histogram2d(self.px/self.dx, self.py/self.dx, bins=[edges_x, edges_y])[0]/(self.dx*self.dx)

		# advection current calculations
		Jadv_x = self.phi*self.ux[:,:self.Ny]
		Jadv_y = self.phi*self.uy[:,:self.Ny]
		divJadv = self.calculate_derivative(Jadv_x, 'wrt_x', True) + self.calculate_derivative(Jadv_y, 'wrt_y', True)

		# calculate mu
		self.mu_phi = -self.B*self.phi + self.B*self.phi*self.phi*self.phi - 0.5*self.B*self.calculate_derivative(self.phi, 'laplacian', False) - self.C*self.psi
                # calculate phi
		self.phi += -self.dt*divJadv + self.dt*self.M_phi*self.calculate_derivative(self.mu_phi, 'laplacian', False)
		
	''' Update particles '''
	def update_particles(self):
		# find the array indes (i,j) for each particle's position
		i = np.round(self.px/self.dx).astype(int)
		j = np.round(self.py/self.dx).astype(int)
		# ensure there are no impossible indexes calculated
		i[i >= self.Nx] = self.Nx - 1
		j[j >= self.Ny] = self.Ny - 1
		i[i < 0] = 0
		j[j < 0] = 0
		
		# noise for brownian motion
		noise = np.random.normal(0.0, 1.0, (2, self.Np))
		
		# calculate gradient of phi
		dphidx = self.calculate_derivative(self.phi, 'wrt_x', False)
		dphidy = self.calculate_derivative(self.phi, 'wrt_y', False)
		
		# update particles' positions
		self.px += self.dt*self.ux[i,j] + np.sqrt(2.0*self.dt)*noise[0] + self.dt*self.C*dphidx[i,j] 
		self.py += self.dt*self.uy[i,j] + np.sqrt(2.0*self.dt)*noise[1] + self.dt*self.C*dphidy[i,j] - self.dt*self.G
		
		# implement wall boundary conditions
		too_low = self.py < -0.5*self.dx
		self.py[too_low] = -0.5*self.dx
		too_high = self.py > self.Ly-0.5*self.dx
		self.py[too_high] = self.Ly-0.5*self.dx

		# implement periodic boundary conditions
		self.px = ((self.px+0.5*self.dx) % self.Lx) - 0.5*self.dx

	''' Update velocity '''
	def update_velocity(self):
		# derivative calculations
		dmuphidx = self.calculate_derivative(self.mu_phi, 'wrt_x', False)
		dmuphidy = self.calculate_derivative(self.mu_phi, 'wrt_y', False)
		dphidx = self.calculate_derivative(self.phi, 'wrt_x', False)
		dphidy = self.calculate_derivative(self.phi, 'wrt_y', False)
		dpsidx = self.calculate_derivative(self.psi, 'wrt_x', False)
		dpsidy = self.calculate_derivative(self.psi, 'wrt_y', False)

		# calculate force density
		fx = np.zeros((self.Nx, 2*self.Ny))
		fy = np.zeros((self.Nx, 2*self.Ny))
		fx[:,:self.Ny] = -self.phi*dmuphidx - dpsidx + self.C*self.psi*dphidx
		fy[:,:self.Ny] = -self.phi*dmuphidy - dpsidy + self.C*self.psi*dphidy - self.G*self.psi
		fx[:,2*self.Ny:self.Ny-1:-1] = -fx[:,:self.Ny]  # impose symmetries
		fy[:,2*self.Ny:self.Ny-1:-1] = -fy[:,:self.Ny]

		# calculate Fourier transform of the force
		fx_ft = np.fft.fft2(fx, norm='ortho')*np.sqrt(self.dx*self.dx)
		fy_ft = np.fft.fft2(fy, norm='ortho')*np.sqrt(self.dx*self.dx)

		# q dot f
		qdotf = self.qx*fx_ft + self.qy*fy_ft

		# calculate Fourier transform of the fluid velocity using Stokes formula
		ux_ft = (fx_ft/self.q2 - self.qx*qdotf/self.q4)/self.eta
		uy_ft = (fy_ft/self.q2 - self.qy*qdotf/self.q4)/self.eta
		ux_ft[0,0] = 0.0
		uy_ft[0,0] = 0.0

		# get fluid velocity using inverse Fourier transform
		self.ux = np.real(np.fft.ifft2(ux_ft, norm='ortho')/np.sqrt(self.dx*self.dx))
		self.uy = np.real(np.fft.ifft2(uy_ft, norm='ortho')/np.sqrt(self.dx*self.dx))
		
	''' Run simulation '''
	def run(self):
		# create new folder to write the data if folder already existed erase data
		try:
			os.mkdir('./data')
		except:
			shutil.rmtree('./data')
			os.mkdir('./data')
			
                # running loop
		for nt in range(0, self.Nt):
			self.update_phi()
			self.update_particles()
			self.update_velocity()
			
			if nt % self.nt_save == 0:
				phi0 = np.sum(self.phi)/(self.Lx*self.Ly)
				print(f'timestep = {nt}; phi0 = {phi0}')
				np.savetxt(f'./data/phi{nt}.txt', self.phi)
				np.savetxt(f'./data/px{nt}.txt', self.px)
				np.savetxt(f'./data/py{nt}.txt', self.py)
				np.savetxt(f'./data/ux{nt}.txt', self.ux)
				np.savetxt(f'./data/uy{nt}.txt', self.uy)	
	
	''' Create animation '''			
	def animate(self):
		# initialize figure and movie objects
		fig, ax = plt.subplots(1, 2, figsize=(12,8))

		# set labels
		for n in range(0, 2):
			ax[n].set_xlabel('$x$')
			ax[n].set_ylabel('$y$')

			ax[n].set_xlim([-0.5*self.dx, self.Lx-0.5*self.dx])
			ax[n].set_ylim([-0.5*self.dx, self.Ly-0.5*self.dx])

			ax[n].set_aspect(1)

		# create colormap of phi
		phi = np.loadtxt(f'./data/phi0.txt')
		graph00 = ax[0].pcolormesh(self.X, self.Y, phi, vmin=-1.2, vmax=1.2, shading='auto')
		fig.colorbar(graph00, ax=ax[0], shrink=0.8)

		phi = np.loadtxt(f'./data/phi0.txt')
		graph10 = ax[1].pcolormesh(self.X, self.Y, phi, vmin=-1.2, vmax=1.2, shading='auto')
		fig.colorbar(graph10, ax=ax[1], shrink=0.8)

		px = np.loadtxt(f'./data/px0.txt')
		py = np.loadtxt(f'./data/py0.txt')
		graph01, = ax[0].plot(px, py, '.', color='black', markersize=2)

		graph11 = ax[1].quiver(self.X[::2,::2], self.Y[::2,::2], self.ux[::2,:self.Ny:2], self.uy[::2,:self.Ny:2], \
			   angles='xy', scale_units='xy', scale=0.01)

		# define a local function animate, which reads data at time step nt, and update the plot
		def animate(nt):
			phi = np.loadtxt(f'./data/phi{nt}.txt')
			graph00.set_array(phi.flatten())  # update data
			graph10.set_array(phi.flatten())  # update data

			px = np.loadtxt(f'./data/px{nt}.txt')
			py = np.loadtxt(f'./data/py{nt}.txt')
			graph01.set_data(px, py)

			ux = np.loadtxt(f'./data/ux{nt}.txt')
			uy = np.loadtxt(f'./data/uy{nt}.txt')
			graph11.set_UVC(ux[::2,:self.Ny:2], uy[::2,:self.Ny:2])

		# interval = time between frames in miliseconds
		anim = animation.FuncAnimation(fig, animate, frames=range(0, self.Nt, self.nt_save), interval = 100, \
												blit = False, repeat = False)  
		anim.save('animation.mp4')  # use .mp4 or .gif if not working
		
if __name__ == '__main__':	
	program = HybridParticle()
	program.run()
	program.animate()

		

