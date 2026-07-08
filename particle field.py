import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import shutil
import cProfile
import pstats

#np.set_printoptions(threshold=np.inf)

class CoffeeRing():

    def __init__(self,name,consts):
        '''Initialise function'''

        self.name = name
        
        try:
            os.mkdir('./'+self.name+'data')
        except OSError:
            shutil.rmtree('./'+self.name+'data')
            os.mkdir('./'+self.name+'data')

        #spatial and time stepping constants
        #when halving dx, dt needs to divide by ten, vice versa
        self.dx = 0.5
        self.halfdx = self.dx/2
        self.dt = 0.001
        
        #spatial and time dimensions
        self.Lx = 64
        self.Ly = 32
        self.Nt = int(25e6)
        self.save = int(1e4)
        self.Nx = int(self.Lx/self.dx)
        self.Ny = int(self.Ly/self.dx)
        
        #initialise the spatial meshgrid
        self.x = np.arange(0.0, self.Lx, self.dx)
        self.y = np.arange(0.0, self.Ly, self.dx)
        self.size = (len(self.x),len(self.y))
        self.Y, self.X = np.meshgrid(self.y, self.x)

        #droplet constants
        self.radius = 20
        self.mp = 9

        #chemical potential constants
        self.alpha = -1.0
        self.beta = 1.0
        self.kappa = 0.25
        self.M_phi = 1.0

        #particle constants
        self.kBT = consts[1]
        self.G = consts[2]
        self.c = consts[0]
        self.M_psi = consts[3]
        self.concentration = consts[4]
        
        #initialise phi and mu arrays
        self.phi = np.zeros(self.size, dtype = 'float64')
        self.psi = np.zeros(self.size,dtype='float64')
        self.mu_phi = np.zeros(self.size,dtype='float64')
        self.init_arrays(False)

    def init_arrays(self,load):
        '''Initialise the phi array to be in the shape of a semi-circle'''

        if load==True:
            path = f'test 10data/phi999000.txt'
            phi = np.loadtxt(path)
            self.phi = phi
            path = f'test 10data/psi999000.txt'
            psi = np.loadtxt(path)
            self.psi = psi  
        else:
            #get indexes of all array slots
            indx, indy = np.where(self.phi==0)

            centre = self.Lx / 2
            #assign each array slot with the effective radius from the centre
            self.phi[indx,indy] = (indx*self.dx - centre)**2 + (indy*self.dx)**2

            drop = self.phi <= self.radius**2
            not_drop = self.phi > self.radius**2

            #assign values
            self.phi[drop] = np.sqrt(-1*self.alpha/self.beta)
            self.phi[not_drop] = -np.sqrt(-1*self.alpha/self.beta)
            self.interface = np.where(self.phi[:,0] == 1)[0][0]
            self.psi[drop] = self.concentration
            self.psi[not_drop] = 0

    def lap(self,q,pin):
        
        #make padded array
        Q = np.zeros((self.size[0]+2,self.size[1]+2))
        Q[1:-1,1:-1] = q
        Q[1:-1,0] = q[:,0]
        Q[1:-1,-1] = q[:,-1]
        Q[0,1:-1] = q[-1,:]
        Q[-1,1:-1] = q[0,:]
        
        if pin:
            mp = 8
            i = self.interface
            Q[1+i-mp:1+i+mp,0]= -1*np.flip(q[i-mp:i+mp,0])
            i = self.Nx - self.interface
            Q[1+i-mp:1+i+mp,0] = -1*np.flip(q[i-mp:i+mp,0])

        devx2 = (Q[2:,1:-1] + Q[0:-2,1:-1])
        devy2 = (Q[1:-1,2:] + Q[1:-1,0:-2])

        return (devx2 + devy2 - 4*q)/(self.dx*self.dx)

    def dev(self,q,neg,pin):

        if neg:
            factor = -1
        else:
            factor = 1

        Q = np.zeros((self.size[0]+2,self.size[1]+2))
        Q[1:-1,1:-1] = q
        Q[1:-1,0] = q[:,0] * factor
        Q[1:-1,-1] = q[:,-1] * factor
        Q[0,1:-1] = q[-1,:]
        Q[-1,1:-1] = q[0,:]
        
        if pin:
            mp = 8
            i = self.interface
            Q[1+i-mp:1+i+mp,0]= -1*np.flip(q[i-mp:i+mp,0])
            i = self.Nx - self.interface
            Q[1+i-mp:1+i+mp,0] = -1*np.flip(q[i-mp:i+mp,0])

        devx = (Q[2:,1:-1] - Q[0:-2,1:-1])
        devy = (Q[1:-1,2:] - Q[1:-1,0:-2])

        return [(devx/(2*self.dx)),(devy/(2*self.dx))]

    def update_phi(self):
        '''function for updating phi'''
        
        self.mu_phi = (self.alpha*self.phi) + (self.beta*self.phi*self.phi*self.phi) - self.c*self.psi - (self.kappa*self.lap(self.phi,1)) 

        self.phi = self.phi + (self.dt * self.M_phi * self.lap(self.mu_phi,0))
        
        self.phi[:,-1] = -np.sqrt(-1*self.alpha/self.beta)
        self.phi[-1,:] = -np.sqrt(-1*self.alpha/self.beta)
        self.phi[0:s-1,:] = -np.sqrt(-1*self.alpha/self.beta)

    def update_psi(self):

        d_phi = self.dev(self.phi,0,1)
        d_psi = self.dev(self.psi,0,0)
        
        Jy = self.G*self.psi + self.kBT*d_psi[1] - self.c*self.psi*d_phi[1]
        Jx = self.kBT*d_psi[0] - self.c*self.psi*d_phi[0] 

        self.psi = self.psi + (self.dt*self.M_psi*(self.dev(Jx,1,0)[0]+ self.dev(Jy,1,0)[1]))
            
    def run(self):
        '''running function'''

        print("STARTING " + self.name)
       
        for nt in range(0, self.Nt, 1):
            self.update_phi()
            self.update_psi()
            # save phi to a file every nt_skip timesteps
            if nt % self.save == 0:
                print(nt)
                #print(np.average(self.psi))
                #print(np.average(self.phi))
                np.savetxt(f'./'+self.name+'data/phi'+str(nt)+'.txt', self.phi)
                np.savetxt(f'./'+self.name+'data/psi'+str(nt)+'.txt', self.psi)

    def plot(self):
        fig, (ax1,ax2) = plt.subplots(2,1)
        
        # set labels
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        # set labels
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')

        # set x range and y range
        ax1.set(xlim=(0,self.Lx), ylim = (0,self.Ly))
        # set x range and y range
        ax2.set(xlim=(0,self.Lx), ylim = (0,self.Ly))

        # set aspect ratio
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        
        # create colormap of phi
        phi = np.loadtxt(f'./'+self.name+'data/phi0.txt')
        graph1 = ax1.pcolormesh(self.X, self.Y, phi, vmin=-1.2, vmax=1.2,shading='auto')
        plt.colorbar(graph1,fraction=0.046,pad=0.04)
        psi = np.loadtxt(f'./'+self.name+'data/psi0.txt')
        graph2 = ax2.pcolormesh(self.X, self.Y, psi,vmin = 0, vmax = 0.2, shading='auto')
        plt.colorbar(graph2,fraction=0.046,pad=0.04)

        # define a local function animate, which reads data at time step nt, and update the plot
        def animate(nt):
            phi = np.loadtxt(f'./'+self.name+'data/phi'+str(nt)+'.txt')
            graph1.set_array(phi.flatten())  # update data
            psi = np.loadtxt(f'./'+self.name+'data/psi'+str(nt)+'.txt')
            graph2.set_array(psi.flatten())  # update data

        # interval = time between frames in miliseconds
        anim = animation.FuncAnimation(fig, animate, frames=range(0, self.Nt, self.save),interval = 300,
                                       blit = False,repeat = False)  
        anim.save((self.name + '.mp4'))

        psi_max = np.loadtxt(f'./'+self.name+'data/psi24990000.txt')
        fig,ax = plt.subplots()
        average_concentration_max = np.mean(psi_max,axis=0)
        N = np.sum(average_concentration_max*self.dx)
        ax.plot(average_concentration_max,self.y)
        print(np.sum(average_concentration_max))
        C = ((self.kBT/self.G)*(1-np.exp(-self.G*self.Ly/self.kBT)))**(-1)
        boltz = np.exp(self.y*-self.G/self.kBT) * N * C 
        print(np.sum(boltz))
        ax.plot(boltz,self.y)
        plt.legend(["max","boltz"])
        fig.savefig(f''+self.name+'boltzmann plot.png')

        print("ENDING " + self.name)

if __name__ == '__main__':

    name = "test"
    consts = [0.05,0.005,0.0005,2,0.05]
    program = CoffeeRing(name,consts)                    
    program.run()
    program.plot()



