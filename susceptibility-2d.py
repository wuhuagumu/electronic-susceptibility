import sys
import numpy as np
from numpy import linalg as LA

def delta_function(x, epsilon=0.00001,method='poisson'):
    if method=='poisson':
        return (1/np.pi)*epsilon/(x**2+epsilon**2)
    elif method=='heat':
        return np.exp(-x**2/(2*epsilon))/np.sqrt(2*np.pi*epsilon)

def fermi_equation(energy, mu=0, T=20):
    k = 8.6173303E-5
    return 1/(np.exp((energy-mu)/(k*T))+1)

def read_CONTCAR(filename):
    file = open(filename,'r')
    file.readline()
    scale = float(file.readline())
    a1 = file.readline().split()
    a2 = file.readline().split()
    a3 = file.readline().split()
    a = [a1, a2, a3]
    a = np.array(a, dtype='float') * scale
    file.close()
    print("Reading processe is done (CONTCAR)")
    return a

def get_b(a):  # this funciton is to get reciprocal lattice from primitive lattice

    v = np.dot(a[0], np.cross(a[1], a[2]))
    b = []
    b.append(2 * np.pi * np.cross(a[1], a[2]) / v)
    b.append(2 * np.pi * np.cross(a[2], a[0]) / v)
    b.append(2 * np.pi * np.cross(a[0], a[1]) / v)
    b = np.array(b, dtype='float')
    return b

def point_scale(pt_list, a):
    # point_scale, including rpt and kpt, if it is rpt, put a as lattice, if it is kpt, put a as inversed lattice
    pt_scaled=[]
    for pt in pt_list:
        pt_scaled.append(np.dot(pt, a))
    pt_scaled = np.array(pt_scaled, dtype='float')
    return pt_scaled

def read_hr(filename):
    file = open(filename,'r')
    file.readline()
    num_wann = int(file.readline().split()[0])
    nrpts = int(file.readline().split()[0])
    weight = []
    for i in range(int(np.ceil(nrpts / 15.0))):
        buffer = file.readline().split()
        weight = weight + buffer
    weight = np.array(weight, dtype='int')
    rpt = []
    hamr = np.zeros((num_wann, num_wann, nrpts), dtype='complex')

    for i in range(nrpts):
        for j in range(num_wann):
            for k in range(num_wann):
                buffer = file.readline().split()
                hamr[k, j, i] = float(buffer[5]) + 1j * float(buffer[6])
        rpt = rpt + [buffer[0:3]]

    rpt = np.array(rpt, dtype='int')
    hr = {'num_wann': num_wann, 'nrpts': nrpts, 'weight': weight, 'rpt': rpt, 'hamr': hamr}
    file.close()
    print("Reading processe is done (hr)")
    return hr

def fourier(kpt_list, hr):

    kpt_list = point_scale(kpt_list, hr['b'])
    rpts = point_scale(hr['rpt'], hr['a'])
    phase = np.exp(1j * np.dot(rpts, kpt_list.T))/hr['weight'][:, None]

    hamk = np.tensordot(hr['hamr'], phase, axes=1)
    #print(np.shape(hr['hamr']), np.shape(hr['hami']), np.shape(rpts), kpoint, np.shape(hr['a']), np.shape(hr['b']))

    return hamk

def gap_grid(kpt_list, hr):

    hamk = fourier(kpt_list, hr)
    hamk = np.rollaxis(hamk, 2, 0)
    print(np.shape(hamk))
    w= LA.eigvalsh(hamk)

    return w

def write_contour(x,y,weight,fn):#fn filename
    m,n=np.shape(np.array(x))
    with open(fn,'w') as f:
        for i in range(m):
            for j in range(n):
                print('%f    %f    %f' % (x[i,j],y[i,j],weight[i,j]),file=f)
    return

def gen_kpt_grid(nk):
    nkpt = nk[0] * nk[1] * nk[2]
    kx = np.linspace(0,1,nk[0], endpoint=False)
    ky = np.linspace(0,1,nk[1], endpoint=False)
    kz = np.linspace(0,1,nk[2], endpoint=False)

    kpt_list = []
    kpt_label = []
    for i in range(nk[0]):
        for j in range(nk[1]):
            for k in range(nk[2]):
                kpt_list.append([kx[i], ky[j], kz[k]])
                kpt_label.append(j)
    return kpt_list

def calc_chi_imag(nk, w_reshaped, ef=0, epsilon='0.01', method='poisson'):
    chi_imag = np.zeros((nk[0], nk[1], nk[2]), dtype = 'float')
    for i in range(nk[0]):
        for j in range(nk[1]):
            for k in range(nk[2]):
                tmp = np.roll(w_reshaped, i, axis=0)
                tmp = np.roll(tmp, j, axis=1)
                tmp = np.roll(tmp, k, axis=2)
                for m in band_cross_ef1:
                    for n in band_cross_ef2:
                        #tmp1 = np.multiply(delta_function(w_reshaped[:,:,:,m]-ef, epsilon=0.05), \
                        #                                     delta_function(tmp[:,:,:,n]-ef, epsilon=0.05))
                        chi_imag[i,j,k] += np.sum(np.multiply(delta_function(w_reshaped[:,:,:,m]-ef, epsilon=epsilon,method=method), \
                                                             delta_function(tmp[:,:,:,n]-ef, epsilon=epsilon,method=method)))
    return chi_imag

def calc_chi_real(nk, w_reshaped, ef=0, T=20, delta=0.001):
    chi_real = np.zeros((nk[0],nk[1],nk[2]), dtype = 'complex')
    for i in range(nk[0]):
        for j in range(nk[1]):
            for k in range(nk[2]):
                tmp = np.roll(w_reshaped, i, axis=0)
                tmp = np.roll(tmp, j, axis=1)
                tmp = np.roll(tmp, k, axis=2)
                for m in band_cross_ef1:
                    for n in band_cross_ef2:
                        chi_real[i,j,k] += np.sum((fermi_equation(w_reshaped[:,:,:,m],mu=ef, T=T)-\
                                                       fermi_equation(tmp[:,:,:,n],mu=ef, T=T))/\
                                                      (w_reshaped[:,:,:,m]-tmp[:,:,:,n]+ 1j * delta))
    return np.real(-chi_real)


######INPUT############
filename_CONTCAR = './CONTCAR'
filename_hr = './cdw1_hr.dat'
ef= -0.1390 # fermi energy
nk = [200,200,1] # number of kpoints
band_cross_ef1 = [0,1,2,3,4,5,6,7,8] # indices of bands that cross Ef in wannier basis [0]
band_cross_ef2 = [0,1,2,3,4,5,6,7,8] # indices of bands that cross Ef in wannier basis [0]
fn_prefix = 'cdw1' # imaginary file name
task=0 # 0 for imag, 1 for real, 2 for both
T = 20 # temperature for real part
delta = 0.000001 # delta for real part
epsilon = 0.01 # epsilon for delta function for imag part
method = 'poisson' # 'poisson or heat' method of delta function for imag part
nboxes = 100 # boxes for contour plot
######END OF INPUT############

hr = read_hr(filename_hr)
hr['a'] = read_CONTCAR(filename_CONTCAR)
hr['b'] = get_b(hr['a'])

# nk = [400,400,1]

kpt_list=gen_kpt_grid(nk)
print("shape of kpt_list", np.shape(kpt_list))
kpt_list = np.array(kpt_list)
kpt_list_scaled = point_scale(kpt_list,hr['b'])
X = kpt_list_scaled[:,0].reshape((nk[0], nk[1], nk[2]))[:,:,0]
Y = kpt_list_scaled[:,1].reshape((nk[0], nk[1], nk[2]))[:,:,0]
# X1 = kpt_list[:,0].reshape((nk[0], nk[1], nk[2]))[:,:,0]
# Y1 = kpt_list[:,1].reshape((nk[0], nk[1], nk[2]))[:,:,0]

w = gap_grid(kpt_list, hr)

# wmin = w.min(axis=0)
# wmax = w.max(axis=0)

# band_cross_ef = []
# for i in range(hr['num_wann']):
#     if wmin[i] <= ef and ef <= wmax[i]:
#         band_cross_ef.append(i)
# print(band_cross_ef)

w_reshaped = w.reshape((nk[0], nk[1], nk[2], hr['num_wann']))


if task==0:
    chi_imag=calc_chi_imag(nk, w_reshaped, ef=ef, epsilon=epsilon, method=method)
    write_contour(X,Y,chi_imag[:,:,0],fn_prefix+'_imag.dat')
    print("imag written")
elif task==1:
    chi_real=calc_chi_real(nk, w_reshaped, ef=ef, T=T, delta=delta)
    write_contour(X,Y,-chi_real[:,:,0],fn_prefix+'_real.dat')
    print("real written")
elif task==2:
    chi_imag=calc_chi_imag(nk, w_reshaped, ef=ef, epsilon=epsilon, method=method)
    write_contour(X,Y,chi_imag[:,:,0],fn_prefix+'_imag.dat')
    print("imag written")
    chi_real=calc_chi_real(nk, w_reshaped, ef=ef, T=T, delta=delta)
    write_contour(X,Y,-chi_real[:,:,0],fn_prefix+'_real.dat')
    print("real written")


################comment######################
try:
    import matplotlib
except ImportError:
    print('no matplotlib')
    sys.exit(1)
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm

if 'chi_imag' in vars():
    plt.contourf(X,Y,np.log(chi_imag[:,:,0]),nboxes)
    plt.axis('equal')
    plt.colorbar()
    plt.savefig('imag.png')
    plt.show()

if 'chi_real' in vars():
    plt.contourf(X, Y, chi_real[:,:,0],nboxes)
    plt.axis('equal')
    plt.colorbar()
    plt.savefig('real.png')
    plt.show()
####END OF PLOT#########
