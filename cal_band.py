import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import numba as nb

with open('TG_hr.dat', 'r') as f:
    lines = f.readlines()
# 提取前四行数据
header_lines = lines[1:4]

# 获得前四行中的后三行的数据
line_num = int(header_lines[0].split()[0])
orbital_num = int(header_lines[1].split()[0])
R_num = int(header_lines[2].split()[0])
# print(line_num, orbital_num, R_num)

l_max = 3  # -1, 0, 1
m_max = 3  # -1, 0, 1
n_max = 1  # 0

# 提取实空间的哈密顿量
H = np.zeros((l_max, m_max, n_max, orbital_num, orbital_num), dtype=np.float64)
for i in range(int(line_num)):
    line = lines[i+5].split()
    l = int(line[0])+1
    m = int(line[1])+1
    n = int(line[2])
    j = int(line[3])-1
    k = int(line[4])-1
    t = float(line[5])
    # print(l,m,n,j,k,t)
    H[l, m, n, j, k] = t


# Set the unitcell vectors
with open('POSCAR', 'r') as f:
    lines = f.readlines()

a = np.array([float(lines[2].split()[0]), float(
    lines[2].split()[1]), float(lines[2].split()[2])])
b = np.array([float(lines[3].split()[0]), float(
    lines[3].split()[1]), float(lines[3].split()[2])])
c = np.array([float(lines[4].split()[0]), float(
    lines[4].split()[1]), float(lines[4].split()[2])])

vol = np.dot(a, np.cross(b, c))
astar = 2*np.pi*np.cross(b, c)/vol
bstar = 2*np.pi*np.cross(c, a)/vol
cstar = 2*np.pi*np.cross(a, b)/vol


# 实空间的哈密顿量转换为倒空间的哈密顿量
@nb.jit(parallel=True, nopython=True, fastmath=True)
def Getk(H, ka, kb, kc):
    mk = np.zeros((orbital_num, orbital_num), dtype=np.complex128)
    for j in range(orbital_num):
        for k in range(orbital_num):
            for l in range(0, l_max):
                l = l - 1
                for m in range(0, m_max):
                    m = m - 1
                    for n in range(0, n_max):
                        mjk = H[l, m, n, j, k]
                        kvec = ka*astar + kb*bstar + kc*cstar
                        Rvec = l*a + m*b + n*c
                        mk[j][k] = mk[j][k] + mjk * \
                            np.exp(-1j*np.dot(kvec, Rvec))
    return mk


# Calculate the reciprocal space vectors
m_g_unitvec_1 = 2*np.pi*np.cross(b, c)/vol
m_g_unitvec_2 = 2*np.pi*np.cross(c, a)/vol

m_g_unitvec_1 = m_g_unitvec_1[:2]
m_g_unitvec_2 = m_g_unitvec_2[:2]

# 高对称点
m_gamma_vec = np.array([0, 0])
m_k1_vec = -(m_g_unitvec_1+m_g_unitvec_2)/3-m_g_unitvec_1/3
m_k2_vec = -(m_g_unitvec_1+m_g_unitvec_2)/3-m_g_unitvec_2/3
m_m_vec = (m_k1_vec+m_k2_vec)/2

# 设置k点网格
nk = 10
num_sec = 4
ksec = np.zeros((num_sec, 2), float)
num_kpt = nk*(num_sec-1)
kline = np.zeros((num_kpt), float)
kmesh = np.zeros((num_kpt, 2), float)

# set k path from gamma to k1, m and gamma
ksec[0] = m_gamma_vec
ksec[1] = m_k1_vec
ksec[2] = m_m_vec
ksec[3] = m_gamma_vec

for i in range(num_sec-1):
    vec = ksec[i+1]-ksec[i]
    klen = np.sqrt(np.dot(vec, vec))
    step = klen/(nk)

    for ikpt in range(nk):
        kline[ikpt+i*nk] = kline[i*nk-1]+ikpt*step
        kmesh[ikpt+i*nk] = vec*ikpt/(nk-1)+ksec[i]

# 设置能带数
emesh = np.zeros((nk*4, orbital_num), dtype=np.float64)
kpath = []

# set kpath from gamma to k1, k2, m and gamma
for ik in range(nk):
    kpath.append(m_gamma_vec + (m_k1_vec-m_gamma_vec)*ik/(nk-1))
for ik in range(nk):
    kpath.append(m_k1_vec + (m_m_vec-m_k1_vec)*ik/(nk-1))
for ik in range(nk):
    kpath.append(m_m_vec + (m_k2_vec-m_m_vec)*ik/(nk-1))

# calculate the band structure
for ik in kpath:
    hk = Getk(H, kpath[ik][0], kpath[ik][1], kpath[ik][2])
    EIG, VEC = LA.eig(hk)
    EIG = np.real(EIG)
    EIG = np.sort(EIG)
    for iband in range(orbital_num):
        emesh[ik][iband] = EIG[iband]

# kline是一个一维数组，存储了k点的坐标，kmesh是一个二维数组，存储了k点的坐标，emesh是一个二维数组，存储了能带的能量
# 画能带图
plt.figure(figsize=(10, 10))
for iband in range(orbital_num):
    plt.plot(kline, emesh[:, iband], 'k-')
plt.xlim(kline[0], kline[-1])
plt.ylim(-3, 3)
plt.xlabel('k')
plt.ylabel('Energy')
plt.show()
