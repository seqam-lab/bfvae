

###############################################################################

#
# oval_dsprites (etaS = 0.5, etaH = 0.5, rseed = 10)
#

#-----------------------------------------------------------------------------#
# learned r (relevance vector), a (prior params), Var(p(alpha)) (prior vars)
#   and Var(q(alpha)) (posterior vars)

import numpy as np
import matplotlib.pyplot as plt

#### r

aa = [0.79, 0.84, 0.,   0.,   0.69, 0.55, 0.,   0.4,  0.,   0.  ]
#aa = [0. ,  0. ,  0.06, 0. ,  0. ,  0.98, 0.99, 0. ,  0.87, 0.  ]

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa)
plt.hlines(1.0, 1, 10, 'r', 'dotted')
plt.xticks(range(1,11), range(1,11))
plt.title('BF-VAE-2: Learned relevance vector r')
plt.ylabel(r'$r_j$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae2_oval_learned_rv.png')
fig.savefig('bfvae2_oval_learned_rv.pdf')

#### a

aa = [   1.27,    1.19, 1001.97,  999.26,    1.46,    1.81, 1001.61,    2.47, 
      1001.99, 1001.95]
#aa = [1002. ,  1002.   ,  16.01 ,1002. ,  1002.  ,    1.03  ,  1.01 , 574.06,
#      1.15, 1002.  ]

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa)
#plt.hlines(1.0, 1, 10, 'r', 'dotted')
plt.xticks(range(1,11), range(1,11))
plt.title('BF-VAE-2: Learned prior params a')
plt.ylabel(r'$a_j$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae2_oval_learned_a.png')
fig.savefig('bfvae2_oval_learned_a.pdf')

#### Var(p(alpha))

aa = np.array([   1.27,    1.19, 1001.97,  999.26,    1.46,    1.81, 1001.61, 
               2.47, 1001.99, 1001.95])
#aa = np.array([1002. ,  1002.   ,  16.01 ,1002. ,  1002.  ,    1.03  ,  1.01 ,
#                574.06, 1.15, 1002.  ])

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa/((aa-1.0)**2))
plt.xticks(range(1,11), range(1,11))
plt.title(r'BF-VAE-2: Variance of prior of $\alpha$, $V(p(\alpha)) = a / b^2$')
plt.ylabel(r'$V(p(\alpha_j))$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae2_oval_learned_vpalpha.png')
fig.savefig('bfvae2_oval_learned_vpalpha.pdf')

#### Var(q(alpha))

aa = np.array([ 72.48 , 72.76, 243.55, 250.7,   50.56, 230.29, 242.11,  58.97,
               253.28, 231.79])

bb = np.array([ 8599.5,  11030.2,    126.03,   130.4,   6740.36,  1840.82,
               127.57,  4773.6, 132.38,   121.29])
#aa = np.array([272.2 , 278.7 ,  71.58, 280.33, 277.12,  43.09,  39.72,  68.31,
#               72.66, 278.44])
#bb = np.array([  153.19,   158.71,  3232.86,   158.11,   157.36, 20756.24, 
#               17205.69,   332.28, 9171.59,   157.26])

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa/(bb**2))
plt.xticks(range(1,11), range(1,11))
plt.title(r'BF-VAE-2: Variance of posterior of $\alpha$, $V(q(\alpha)) = \hat{a} / {\hat{b}}^2$')
plt.ylabel(r'$V(q(\alpha_j))$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae2_oval_learned_vqalpha.png')
fig.savefig('bfvae2_oval_learned_vqalpha.pdf')


#-----------------------------------------------------------------------------#
# individual expected kl divs 

import numpy as np
import matplotlib.pyplot as plt

aa = [3.10214459e+00, 3.05938477e+00, 1.34891418e-03, 1.28913002e-03,
 5.01067442e+00, 3.64484587e+00, 1.19931189e-03, 5.03756590e+00,
 1.27045655e-03, 1.31158416e-03]
#aa = [4.75012975e-03, 1.21582898e-03, 2.90813104e+00, 1.21149951e-03,
# 9.95059485e-04, 5.09308540e+00, 4.99686356e+00, 3.64199188e+00,
# 3.21298582e+00, 1.58139799e-03]

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa)
plt.xticks(range(1,11), range(1,11))
plt.title('BF-VAE-2: Individual expected prior KL divs')
plt.ylabel(r'$E_{q(\alpha)}E_x[KL(q(z_j|x)||p(z_j))$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae2_oval_indiv_kls.png')
fig.savefig('bfvae2_oval_indiv_kls.pdf')


###############################################################################



# import numpy as np
# import matplotlib.pyplot as plt

# aa = [1.78594780e+01, 6.99166490e+01, 5.55234006e+00, 8.82018681e+00, \
#  1.08942561e-03, 6.84717997e-04, 6.67698989e-04, 9.00873028e-05, \
#  6.22588982e-04, 1.02864478e-03]  # see below

# aa = [5.39261446e+00, 3.59564379e-05, 3.20556870e-01, 6.26621340e+00, \
#  2.28962749e-05, 5.48028565e+00, 2.58204058e-05, 6.24030965e+00, \
#  2.37950310e-05, 1.36690850e-05]

# fig = plt.figure(figsize=(6,2))
# #plt.plot(aa, 'r*:')
# plt.bar(range(1,11), aa)
# plt.xticks(range(1,11), range(1,11))
# plt.title('Vanilla VAE (beta=1.0)')
# plt.ylabel('E_x[ KL(q(z_j|x)||N(0,1)) ]')
# plt.xlabel('j in z_j')
# plt.show()

# fig.savefig('vanilla_vae_kl_inds.png')
# fig.savefig('vanilla_vae_kl_inds.pdf')


# #---------------------------------------------#

# fname = '/home/mikim/Codes/new_vae/rfvae_learn_again/records/tmp.txt'
       
# # read the file
# with open(fname, "r") as fin:
#     rvs = []
#     i = 0
#     for line in fin:
#         txt = line.rstrip()
#         if '[iter 300000' in txt:
#             break
#         if ('metric1' in txt) or ('********' in txt): 
#             i = 0
#             continue
#         i += 1
#         if i==3:  # complete reading one block (of 3 lines)
#             rvs.append( np.fromstring(txt[10:-1], sep=' ') )
#             i = 0
        
# rvs = np.stack(rvs, axis=0)

# fig = plt.figure(figsize=(6,2))
# #plt.plot(aa, 'r*:')
# plt.bar(range(1,11), rvs[-1,:])
# plt.xticks(range(1,11), range(1,11))
   
# fig.savefig('3dfaces_rv_learned.png')
# fig.savefig('3dfaces_rv_learned.pdf')


# #====================================

# # vanilla-vae
# # [1.99161136e-04, 2.98211599e-04, 5.06021077e+00, 5.71163256e-04, \
# #  4.19024558e+00, 3.49458683e+00, 3.73813842e+00, 2.07519055e-04, \
# #  3.67023604e-04, 4.60299872e+00]


# # factor-vae
# # [1.54604065e-03, 5.90471470e-03, 3.70225299e+00, 8.75577014e-04, \
# #  4.99538746e+00, 3.91780720e-01, 4.99382173e+00, 2.41349193e+00, \
# #  1.64643127e-03, 3.03303608e+00]


# # rfvae
# # [1.78594780e+01, 6.99166490e+01, 5.55234006e+00, 8.82018681e+00, \
# #  1.08942561e-03, 6.84717997e-04, 6.67698989e-04, 9.00873028e-05, \
# #  6.22588982e-04, 1.02864478e-03]



# ###############################################################################

# #
# # releveance vector ("r") evolution during training
# #

# import numpy as np
# import matplotlib.pyplot as plt

# #### read record file

# fname = '/home/mikim/Codes/new_vae/rfvae_learn_again/records/tmp.txt'
       
# # read the file
# with open(fname, "r") as fin:
#     rvs = []
#     i = 0
#     for line in fin:
#         txt = line.rstrip()
#         if '[iter 300000' in txt:
#             break
#         if ('metric1' in txt) or ('********' in txt): 
#             i = 0
#             continue
#         i += 1
#         if i==3:  # complete reading one block (of 3 lines)
#             rvs.append( np.fromstring(txt[10:-1], sep=' ') )
#             i = 0
        
# rvs = np.stack(rvs, axis=0)


# #### dim-wise evolution of rvec

# fig = plt.figure(figsize=(18,8))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.plot(50*np.arange(1,6000), rvs[:,i])
#     plt.title('dim' + str(i+1))
# plt.show()
# # fig.savefig('rfvae_learn_rvec.png')
# # fig.savefig('rfvae_learn_rvec.pdf')


# #### animated gif

# everyKframes = 30

# for i in range(0,6000,everyKframes):
#     fig = plt.figure(figsize=(6,2))
#     plt.bar(range(1,11), rvs[i,:])
#     plt.xticks(range(1,11), range(1,11))
#     plt.title('Evolution of r (iter: %d)' % (i*50))
#     plt.ylabel('r_j')
#     plt.ylim([0,1])
#     plt.xlabel('j in r_j')
#     #plt.show()
#     fig.savefig('tmp/rvec_%05d.png' % i)
#     #fig.savefig('tmp/rvec_%05d.pdf' % i)
#     plt.close(fig)

# # make animated gif
# import os, imageio
# DIR = '/home/mikim/Codes/new_vae/rfvae_learn_again/tmp'
# fnames = []
# for i in range(0,6000,everyKframes):
#     f = 'rvec_%05d.png' % i
#     fnames.append(str(os.path.join(DIR, f)))

# images = []
# for filename in fnames:
#     images.append(imageio.imread(filename))

# imageio.mimsave(
#     '/home/mikim/Codes/new_vae/rfvae_learn_again/tmp/anim_rvec.gif', 
#     images, duration=0.18 )
