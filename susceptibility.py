import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

filename_CONTCAR = 'CONTCAR'
filename_hr = 'wannier90_hr.dat'
nk = [20, 20, 10]
ef = 5.8257
band_cross_ef1 = [30]
band_cross_ef2 = [30]
direction = [1, 2, 2]
position = [0, 0, 5]
T = 20
epsilon = 0.05
delta = 0.0000001


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


def gap_grid(kpt_list, hr):
    hamk = fourier(kpt_list, hr)
    hamk = np.rollaxis(hamk, 2, 0)
    print('gap_grid', np.shape(hamk))
    w = LA.eigvalsh(hamk)

    return w


hr = read_hr(filename_hr)
hr['a'] = read_CONTCAR(filename_CONTCAR)
hr['b'] = get_b(hr['a'])

nkpt = nk[0] * nk[1] * nk[2]
kx = np.linspace(0, 1, nk[0], endpoint=False)
ky = np.linspace(0, 1, nk[1], endpoint=False)
kz = np.linspace(0, 1, nk[2], endpoint=False)

kpt_list = []
kpt_label = []
for i in range(nk[0]):
    for j in range(nk[1]):
        for k in range(nk[2]):
            kpt_list.append([kx[i], ky[j], kz[k]])
            kpt_label.append(j)
w = gap_grid(kpt_list, hr)

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


w_reshaped = w.reshape((nk[0], nk[1], nk[2], hr['num_wann']))

chi_imag = np.zeros((nk[0], nk[1], nk[2]), dtype='float')
chi_real = np.zeros((nk[0], nk[1], nk[2]), dtype='complex')
for i in range(nk[0]):
    tmp = np.roll(w_reshaped, i, axis=0)
    for j in range(nk[1]):
        tmp = np.roll(tmp, j, axis=1)
        for k in range(nk[2]):
            tmp = np.roll(tmp, k, axis=2)
            for m in band_cross_ef1:
                for n in band_cross_ef2:
                    chi_imag[i, j, k] += np.sum(
                        np.multiply(delta_function(w_reshaped[:, :, :, m] - ef, epsilon=epsilon),
                                    delta_function(tmp[:, :, :, n] - ef, epsilon=epsilon)))
                    chi_real[i, j, k] += np.sum((fermi_equation(w_reshaped[:, :, :, m], mu=ef, T=T) -
                                                 fermi_equation(tmp[:, :, :, n], mu=ef, T=T)) / \
                                                (w_reshaped[:, :, :, m] - tmp[:, :, :, n] + 1j * delta))
    print(i)

chi_real = -np.real(chi_real)  # notice that here has a minus sign
kpt_list = np.array(kpt_list)
kpt_list_scaled = point_scale(kpt_list, hr['b'])
'''
X = kpt_list_scaled[:, 0].reshape((nk[0], nk[1], nk[2]))[:, 0, :]
Y = kpt_list_scaled[:, 1].reshape((nk[0], nk[1], nk[2]))[:, :, :]
Z = kpt_list_scaled[:, 2].reshape((nk[0], nk[1], nk[2]))[:, 0, :]
X1 = kpt_list[:, 0].reshape((nk[0], nk[1], nk[2]))[:, 0, :]
Y1 = kpt_list[:, 1].reshape((nk[0], nk[1], nk[2]))[:, :, :]
Z1 = kpt_list[:, 2].reshape((nk[0], nk[1], nk[2]))[:, 0, :]
# print(kx[:,:,0:1])
plt.contourf(X1, Z1, chi_imag[:, 0, :], 50)
plt.colorbar()
plt.show()
# plt.axis([0,3.5,0,3.5])
# plt.axis('equal')
'''

X1 = kpt_list[:, 0].reshape((nk[0], nk[1], nk[2]))[:, :, :]
Y1 = kpt_list[:, 1].reshape((nk[0], nk[1], nk[2]))[:, :, :]
Z1 = kpt_list[:, 2].reshape((nk[0], nk[1], nk[2]))[:, :, :]

for i in range(len(direction)):
    k1 = (direction[i] + 1) % 3
    k2 = (direction[i] + 2) % 3
    print(direction[i], k1, k2, nk[k1], nk[k2],position[i])
    fn = open('susceptibility-' + str(direction[i]) + '-' + str(position[i]) + '.dat', 'w')
    if direction[i] == 0:
        print('k_x' + '=' + str(kx[position[i]]), 'band cross ef', band_cross_ef1, band_cross_ef2, file=fn)
        for j in range(nk[k1]):
            for k in range(nk[k2]):
                line = '{0:8f}    {1:8f}    {2:8f}    {3:8f}    {4:8f}'.format(
                    X1[position[i], j, k],
                    Y1[position[i], j, k],
                    Z1[position[i], j, k],
                    chi_real[position[i], j, k],
                    chi_imag[position[i], j, k])
                print(line, file=fn)
    elif direction[i] == 1:
        print('k_y' + '=' + str(ky[position[i]]), 'band cross ef', band_cross_ef1, band_cross_ef2, file=fn)
        for j in range(nk[k1]):
            for k in range(nk[k2]):
                line = '{0:8f}    {1:8f}    {2:8f}    {3:8f}    {4:8f}'.format(
                    X1[k, position[i], j],
                    Y1[k, position[i], j],
                    Z1[k, position[i], j],
                    chi_real[k, position[i], j],
                    chi_imag[k, position[i], j])
                print(line, file=fn)
    elif direction[i] == 2:
        print('k_z' + '=' + str(kz[position[i]]), 'band cross ef', band_cross_ef1, band_cross_ef2, file=fn)
        for j in range(nk[k1]):
            for k in range(nk[k2]):
                line = '{0:8f}    {1:8f}    {2:8f}    {3:8f}    {4:8f}'.format(
                    X1[j, k, position[i]],
                    Y1[j, k, position[i]],
                    Z1[j, k, position[i]],
                    chi_real[j, k, position[i]],
                    chi_imag[j, k, position[i]])
                print(line, file=fn)
    else:
        print('ERROR of direction', i)

    fn.close()
