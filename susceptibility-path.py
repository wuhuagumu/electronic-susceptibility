import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

filename_CONTCAR = 'CONTCAR'
filename_hr = 'wannier90_hr.dat'
nk = [20, 20, 10]
ef = 5.8257
band_cross_ef1 = [30]
band_cross_ef2 = [30]
T = 20
epsilon = 0.05
delta = 0.0000001
qpath = [[0,0,0],[0.5,0,0]]
nq = 5



def delta_function(x, epsilon=0.00001):
    return (1 / np.pi) * epsilon / (x ** 2 + epsilon ** 2)


def fermi_equation(energy, mu=0, T=20):
    k = 8.6173303E-5
    return 1 / (np.exp((energy - mu) / (k * T)) + 1)


def read_CONTCAR(filename):
    file = open(filename, 'r')
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
    pt_scaled = []
    for pt in pt_list:
        pt_scaled.append(np.dot(pt, a))
    pt_scaled = np.array(pt_scaled, dtype='float')
    return pt_scaled


def point_reverse_scale(pt_list, a):
    pt_reverse_scale = []
    for pt in pt_list:
        pt_reverse_scale.append([LA.solve(a, pt)])
    pt_reverse_scale = np.array(pt_reverse_scale, dtype='float')
    return pt_reverse_scale


def read_hr(filename):
    file = open(filename, 'r')
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
    phase = np.exp(1j * np.dot(rpts, kpt_list.T)) / hr['weight'][:, None]

    hamk = np.tensordot(hr['hamr'], phase, axes=1)
    # print(np.shape(hr['hamr']), np.shape(hr['hami']), np.shape(rpts), kpoint, np.shape(hr['a']), np.shape(hr['b']))

    return hamk


def eige_batch(kpt_list, hr):
    hamk = fourier(kpt_list, hr)
    hamk = np.rollaxis(hamk, 2, 0)
    print('eige_batch', np.shape(hamk))
    w = LA.eigvalsh(hamk)

    return w

def construct_rpath(num, rpath):  # num per line; rpath could be kpath or qpath, r stand for reciprocal
    if isinstance(num,int):
        num=[num,num,num]
    if len(rpath)-1>len(num):
        raise Error('num and rpath sections not consistent')
    print(num)
    rx = []
    ry = []
    rz = []
    for i in range(len(rpath) - 1):
        rx += list(np.linspace(rpath[i][0], rpath[i + 1][0], num[i], endpoint=False))
        ry += list(np.linspace(rpath[i][1], rpath[i + 1][1], num[i], endpoint=False))
        rz += list(np.linspace(rpath[i][2], rpath[i + 1][2], num[i], endpoint=False))
    rlist = []
    for i in range(len(rx)):
        rlist.append([rx[i], ry[i], rz[i]])
    rlist.append(list(rpath[-1]))
    return rlist


def construct_rgrid(num,vertex=[[0,0,0],[1,1,1]],shift = [0,0,0]): 
    vertex = np.array(vertex)
    n = num[0] * num[1] * num[2]
    rx = np.linspace(vertex[0,0], vertex[1,0], num[0], endpoint=False) + shift[0]
    ry = np.linspace(vertex[0,1], vertex[1,1], num[1], endpoint=False) + shift[1]
    rz = np.linspace(vertex[0,2], vertex[1,2], num[2], endpoint=False) + shift[2]
    rlist = []
    for i in range(num[0]):
        for j in range(num[1]):
            for k in range(num[2]):
                rlist.append([rx[i], ry[j], rz[k]])
    return rlist

def dist_rpath(rlist,a=np.eye(3)):
    pt_list = point_scale(rlist,a)
    dist = [0.0]
    for i in range(len(rlist)-1):
        dist.append(np.linalg.norm(pt_list[i+1]-pt_list[i]))
    return np.cumsum(dist)

hr = read_hr(filename_hr)
hr['a'] = read_CONTCAR(filename_CONTCAR)
hr['b'] = get_b(hr['a'])

nkpt = nk[0] * nk[1] * nk[2]
kpt_list = construct_rgrid(nk)

w = eige_batch(kpt_list, hr)
w_reshaped = w.reshape((nk[0], nk[1], nk[2], hr['num_wann']))
wmin = w.min(axis=0)
wmax = w.max(axis=0)

band_cross_ef = []
for i in range(hr['num_wann']):
    if wmin[i] <= ef and ef <= wmax[i]:
        band_cross_ef.append(i)
print(band_cross_ef)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

qlist = construct_rpath(nq,qpath)
qlist = np.array([[0.4,0,0.5],[0.45,0,0.5]])
print("qlist: ",qlist)
chi_imag = np.zeros(len(qlist))
chi_real = np.zeros(len(qlist))
for iq in range(len(qlist)):
    kq = construct_rgrid(nk,shift=qlist[iq])
    wq = eige_batch(kq, hr)
    wq_reshaped = wq.reshape((nk[0], nk[1], nk[2], hr['num_wann']))
    for m in band_cross_ef1:
        for n in band_cross_ef2:
            #chi_imag[iq] = np.sum(np.multiply(delta_function(w_reshaped[:, :, :, m] - ef, epsilon=epsilon), delta_function(wq_reshaped[:, :, :, n] - ef, epsilon=epsilon)))
            chi_imag[iq] = np.sum(delta_function(w[ :, m] - ef, epsilon=epsilon) * delta_function(wq[:, n] - ef, epsilon=epsilon))
            chi_real[iq] += np.sum((fermi_equation(w[ :, m], mu=ef, T=T) -fermi_equation(wq[:, n], mu=ef, T=T)) / (w[ :, m] - wq[ :, n] + 1j * delta))

qd = dist_rpath(qlist,hr['b'])
print(chi_real,chi_imag)
fn = open('chi-path.dat', 'w')
for iq in range(len(qlist)):
    line = '{0:8f}    {1:8f}    {2:8f}'.format(qd[iq], chi_real[iq], chi_imag[iq])
    print(line, file=fn)
fn.close()

np.multiply(np.array([[1,0],[0,-1]]),np.array([[0,1],[1,0]]))