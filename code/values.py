import scipy.io

mat = scipy.io.loadmat('values.mat')


reconstructions = mat['recon']

og = mat['og']


print(reconstructions[0:4,0,2])
print(og[0:4,0,2])