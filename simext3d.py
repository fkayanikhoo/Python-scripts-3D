###############################################################################
# Read simextNNNN.dat and simphiavgextNNNN.dat to calculate
# Trphi in the mean fluid frame defined by the phi averaged velocity
# routines which work on 3d data are in this file
###############################################################################

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
csv_delimiter = ' '

def open_with_pandas_read_csv(filename):
    df = read_csv(filename, sep=csv_delimiter, header=None)
    data = df.values
    return data

class simext3d():
    ####################################
    # sorting routines so that
    # quantities will be on an indexable
    # grid q_grid[i,j,k]
    # quantities will be accessible as simext3d.q_grid
    # for quantity named q
    ####################################
    def sort_scalar2grid(self,quantity):
        #indice range
        imin = int(np.amin(self.i))
        jmin = int(np.amin(self.j))
        kmin = int(np.amin(self.k))

        imax = int(np.amax(self.i))
        jmax = int(np.amax(self.j))
        kmax = int(np.amax(self.k))

        grid = np.zeros((imax-imin+1, jmax-jmin+1,kmax-kmin+1))
        for i,j,k,q in zip(self.i, self.j, self.k, getattr(self, quantity)):
            grid[int(i)-imin,int(j)-jmin,int(k)-kmin] = q
        setattr(self, quantity+'_grid', grid)
        return

    # works for 4 dimensional vector
    # vector_grid will have shape (Ni, Nj, Nk, 4)
    def sort_vector2grid(self,quantity):
        imin = int(np.amin(self.i))
        jmin = int(np.amin(self.j))
        kmin = int(np.amin(self.k))

        imax = int(np.amax(self.i))
        jmax = int(np.amax(self.j))
        kmax = int(np.amax(self.k))

        grid = np.zeros((imax-imin+1, jmax-jmin+1,kmax-kmin+1,4))
        for i,j,k,q in zip(self.i, self.j, self.k, getattr(self, quantity)):
            grid[int(i)-imin,int(j)-jmin,int(k)-kmin,:] = q
        setattr(self, quantity+'_grid', grid)
        return

    #works for 4x4 D tensor
    # tensor_grid will have shape (Ni, Nj, Nk, 4, 4)
    def sort_tensor2grid(self,quantity):
        imin = int(np.amin(self.i))
        jmin = int(np.amin(self.j))
        kmin = int(np.amin(self.k))

        imax = int(np.amax(self.i))
        jmax = int(np.amax(self.j))
        kmax = int(np.amax(self.k))

        grid = np.zeros((imax-imin+1, jmax-jmin+1,kmax-kmin+1,4,4))
        for i,j,k,q in zip(self.i, self.j, self.k, getattr(self, quantity)):
            grid[int(i)-imin,int(j)-jmin,int(k)-kmin,:,:] = q
        setattr(self, quantity+'_grid', grid)
        return

    ##############################
    # initialize the data structure
    # reads required columns
    ##############################
    def __init__(self, filename, radiation = True, tracer=False):
        # read the simfile
        #data = np.loadtxt(filename)
        data = open_with_pandas_read_csv(filename)
        # macro to select columns as they are labeled
        # in fileop.c
        column = lambda i: data[:,i-1]

        #indices
        self.i = column(1)
        self.j = column(2)
        self.k = column(3)

        #coordinates
        self.r = column(4)
        self.th = column(5)
        self.ph = column(6)

        #density and internal energy
        self.rho = column(7)
        self.uint = column(8)

        # contravariant (upper) componenets of four velocity
        # saved into an array of vectors
        self.ucon = np.zeros((len(self.i),4))
        for i in range(4):
            self.ucon[:,i] = column(9+i)
        metric = calc_schwarzschild_metric_tensor(self.r,self.th)
        tmp = np.einsum('ai,aj,aij->a',self.ucon,self.ucon,metric)
        self.ucon /= np.sqrt(-tmp[:,np.newaxis])
        
        self.bcon = np.zeros((len(self.i),4))
        for i in range(3):
            self.bcon[:,i+1] = column(15+i)

        # stress-energy tensor ^mu _nu
        # saved into an array of tensors
        self.Tmunu = np.zeros((len(self.i),4,4))
        for i in range(4):
            for j in range(4):
                self.Tmunu[:,i,j] = column(18+i*4+j)
#fluid frame magnetic energy density
        #fluid frame radiative energy density
        try:
            self.bsq = column(14)
            self.bcon = np.zeros((len(self.i),4))
            for i in range(3):
                self.bcon[:,i+1] = column(15+i)


        except:
            self.bsq = np.zeros(column(1).shape)


        if tracer:
            try:
                if radiation:
                    self.tracer = column(72)
                else:
                    self.tracer = column(34)
            except:
                self.tracer = np.full(column(1).shape, 1.)


        # radiation stress-energy tensor R^mu _nu
        # saved into an array of tensors
        if radiation:
            self.Ehat = column(34)
            # radiation stress-energy tensor ^mu _nu
            # saved into an array of tensors
            self.Rmunu = np.zeros((len(self.i),4,4))
            for i in range(4):
                for j in range(4):
                    self.Rmunu[:,i,j] = column(35+i*4+j)


            self.Gmu = np.zeros((len(self.i),4))
            for i in range(4):
                self.Gmu[:,i] = column(51 + i)

        self.Titmag = np.zeros(self.ucon.shape)
        self.Titkin = np.zeros(self.ucon.shape)
        self.Titint = np.zeros(self.ucon.shape)
        self.Ehatucon = np.zeros(self.ucon.shape)
        self.rhoucon = np.zeros(self.ucon.shape)

        for i in range(3):
            self.Ehatucon[:,1+i] = column(57+i)
        for i in range(4):
            self.Titmag[:,i] = column(60+i)
            self.Titkin[:,i] = column(64+i)
            self.Titint[:,i] = column(68+i)
            try:
                self.rhoucon[:,i] = column(73+i)
            except:
                self.rhoucon[:,i] = 0.
        self.bsq = column(14)
        self.Ehat = column(34)


# reads a grid r[i,j,k], theta[i,j,k] and computes the metric
# components at each gridpoint
# for 3D data, the metric array will be an
# Ni x Nj x Nk x 4 x 4 dimensional array
def calc_schwarzschild_metric_tensor_on_grid(r_grid,th_grid):
    r = r_grid
    th = th_grid
    ir, ith, iph = r.shape
    metric_tensor = np.zeros((ir,ith,iph,4,4))
    metric_tensor[:,:,:,0,0] = -(1.-2./r)
    metric_tensor[:,:,:,1,1] = 1./(1.-2./r)
    metric_tensor[:,:,:,2,2] = r*r
    metric_tensor[:,:,:,3,3] = r*r*np.sin(th)*np.sin(th)
    return metric_tensor

def calc_schwarzschild_metric_tensor(r_grid,th_grid):
    r = r_grid
    th = th_grid
    ir,= r.shape
    metric_tensor = np.zeros((ir,4,4))
    metric_tensor[:,0,0] = -(1.-2./r)
    metric_tensor[:,1,1] = 1./(1.-2./r)
    metric_tensor[:,2,2] = r*r
    metric_tensor[:,3,3] = r*r*np.sin(th)*np.sin(th)
    return metric_tensor

def calc_inverse_schwarzschild_metric_tensor_on_grid(r_grid,th_grid):
    r = np.ma.masked_where(r_grid<2., r_grid)
    th = th_grid
    ir, ith, iph  = r.shape
    metric_tensor = np.zeros((ir,ith, iph,4,4))
    metric_tensor[:,:,:,0,0] = -1./(1.-2./r)
    metric_tensor[:,:,:,1,1] = (1.-2./r)
    metric_tensor[:,:,:,2,2] = 1./(r*r)
    metric_tensor[:,:,:,3,3] = 1./(r*r*np.sin(th)*np.sin(th))
    return metric_tensor

# compute the lorentz boots matrix from frames.c
# this was just copied from Koral and translated into
# python, the inputs are the metric_tensor_array
# the inverse metric tensor array, and an array of
# the contravariant componenets of the fluid four velocity
def calc_Lorentz_lab2ff_on_grid(metric_tensor, inverse_metric_tensor, ucon_grid):
    # four velocity
    ucon = ucon_grid #shortcut
    #four velocity with lower index
    ucov = np.einsum('ijklm,ijkm->ijkl', metric_tensor, ucon) ## lower index

    # four velocity of the stationary observer (lab frame)
    # wcov = (sqrt(-1/g^tt),0,0,0)
    alpha = np.sqrt(-1./inverse_metric_tensor[:,:,:,0,0])
    wcov = np.zeros(ucon_grid.shape) ## four velocity shape
    wcov[:,:,:,0] = -alpha
    #contravariant components
    wcon = np.einsum('ijklm,ijkm->ijkl',inverse_metric_tensor, wcov)  ## raise index

    #Om is the difference of two outer products
    Om = (np.einsum('ijkl,ijkm->ijklm', ucon, wcov) -
            np.einsum('ijkl,ijkm->ijklm', wcon, ucov) )

    #lorentz factor -w^mu u_mu between lab and fluid frame
    gamma = -np.einsum('ijkl,ijkl->ijk',wcon,ucov)

    #Om sum is omega contracted with itself?
    Omsum = np.einsum('ijklm,ijkmn->ijkln', Om, Om)

    # general expression for lorentz boost
    Lorentz = np.zeros(metric_tensor.shape)
    Lorentz += np.diag([1,1,1,1]) #kronecker delta in koral
    Lorentz += 1./(1.+gamma[:,:,:,np.newaxis, np.newaxis])*Omsum
    Lorentz += Om

    return Lorentz

def calc_onff2lab_transformation_matrix_penna(metric_tensor, ucon):
    gtt = metric_tensor[:,:,:,0,0]
    grr = metric_tensor[:,:,:,1,1]
    gthth = metric_tensor[:,:,:,2,2]
    gphph = metric_tensor[:,:,:,3,3]

    gtph = metric_tensor[:,:,:,0,3]

    ucov = np.einsum('abcij,abci->abcj',metric_tensor, ucon)
    C0 = ucon[:,:,:,0]*ucov[:,:,:,0] + ucon[:,:,:,3]*ucov[:,:,:,3]
    C1 = ucon[:,:,:,1]*ucov[:,:,:,0]
    C2 = ucon[:,:,:,1]*ucov[:,:,:,3]

    ell = ucov[:,:,:,3]/ucov[:,:,:,0]

    N1 = grr*np.sqrt(gtt*C1**2 + grr*C0**2 + gphph*C2**2 + 2.*gtph*C1*C2)
    N2 = np.sqrt(gthth*(1+ucon[:,:,:,2]*ucov[:,:,:,2]))
    N3 = np.sqrt(gtt*ell**2 - 2*gtph*ell - gphph)

    s = -C0/np.fabs(C0)

    eij = np.zeros(metric_tensor.shape)

    #e^i_t
    eij[:,:,:,:,0] = ucon[:,:,:,:]

    #e^t_r
    eij[:,:,:,0,1] = ucov[:,:,:,1]*ucon[:,:,:,0]
    #e^r_r
    eij[:,:,:,1,1] = 1+ucon[:,:,:,1]*ucov[:,:,:,1] + ucon[:,:,:,2]*ucov[:,:,:,2]
    #e^th_r
    eij[:,:,:,2,1] = 0.
    #e^ph_r
    eij[:,:,:,3,1] = ucov[:,:,:,1]*ucon[:,:,:,3]

    #normalize
    eij[:,:,:,:,1] *= s/N1

    #e^t_th
    eij[:,:,:,0,2] = ucov[:,:,:,2]*ucon[:,:,:,0]
    #e^r_th
    eij[:,:,:,1,2] = ucov[:,:,:,2]*ucon[:,:,:,1]
    #e^th_th
    eij[:,:,:,2,2] = 1 + ucov[:,:,:,2]*ucon[:,:,:,2]
    #e^ph_ph
    eij[:,:,:,3,2] =  ucov[:,:,:,3]*ucon[:,:,:,2]

    #normalize
    eij[:,:,:,:,2] *= 1./N2

    #e^t_ph
    eij[:,:,:,0,3] = -ell
    eij[:,:,:,1,3] = 0.
    eij[:,:,:,2,3] = 0.
    eij[:,:,:,3,3] = 1.

    #normalize
    eij[:,:,:,:,3] *= 1./N3

    return eij



# applies the lorentz boost array
# to a contravariant vector
# everything is construct to work with
# data on the grid
# copied from koral/frames.c
def boost_vector_lab2ff_on_grid(vector, inverse_metric_tensor, Lorentz):
    boosted_vector = np.einsum('ijklm,ijkm->ijkl', Lorentz, vector)
    ## make orthnormal?
    #alpha = np.sqrt(-1./inverse_metric_tensor[:,:,:,0,0])
    #return boosted_vector*alpha[:,:,:,np.newaxis]
    return boosted_vector

# same as boos_vector_lab2ff_on_grid but for tensors
# should work for double contra variant tensor (both indices up)
# copied from koral/frames.c
def boost_tensor_lab2ff_on_grid(tensor, inverse_metric_tensor, Lorentz):

    boosted = np.einsum("abcik,abcjl,abckl->abcij", Lorentz, Lorentz, tensor)
    ## make orthnormal?
    #alpha = np.sqrt(-1./inverse_metric_tensor[:,:,:,0,0])
    #boosted[:,:,:,:,0] *= alpha[:,:,:,np.newaxis]
    #boosted[:,:,:,0,:] *= alpha[:,:,:,np.newaxis]
    return boosted






