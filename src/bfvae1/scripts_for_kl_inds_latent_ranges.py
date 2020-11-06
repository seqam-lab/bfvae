

###############################################################################

#
# oval_dsprites (rseed = 11)
#

#-----------------------------------------------------------------------------#
# learned a (prior params), Var(p(alpha)) (prior variances)
#   and Var(q(alpha)) (posterior variances)

import numpy as np
import matplotlib.pyplot as plt

#### a

aa = [1.66, 1.02, 1.02, 1.24, 1.25, 1.03, 1.02, 1.02, 1.33, 1.28]

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa)
plt.hlines(1.0, 1, 10, 'r', 'dotted')
plt.xticks(range(1,11), range(1,11))
plt.title('BF-VAE-1: Learned prior params a')
plt.ylabel(r'$a_j$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae1_oval_learned_a.png')
fig.savefig('bfvae1_oval_learned_a.pdf')

#### Var(p(alpha))

aa = np.array([1.66, 1.02, 1.02, 1.24, 1.25, 1.03, 1.02, 1.02, 1.33, 1.28])

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa/((aa-1.0)**2))
plt.xticks(range(1,11), range(1,11))
plt.title(r'BF-VAE-1: Variance of prior of $\alpha$, $V(p(\alpha)) = a / b^2$')
plt.ylabel(r'$V(p(\alpha_j))$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae1_oval_learned_vpalpha.png')
fig.savefig('bfvae1_oval_learned_vpalpha.pdf')

#### Var(q(alpha))

aa = np.array([ 80.62, 544.52, 492.36, 132.26,  58.84, 577.2,  490.03, 578.19,
               48.96, 66.17 ])
bb = np.array([ 457.39,   12.51,   10.46, 6470.9,  2668.25,   19.15,   10.96,
               13.48, 963.87, 2104.67 ])

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa/(bb**2))
plt.xticks(range(1,11), range(1,11))
plt.title(r'BF-VAE-1: Variance of posterior of $\alpha$, $V(q(\alpha)) = \hat{a} / {\hat{b}}^2$')
plt.ylabel(r'$V(q(\alpha_j))$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae1_oval_learned_vqalpha.png')
fig.savefig('bfvae1_oval_learned_vqalpha.pdf')


#-----------------------------------------------------------------------------#
# individual expected kl divs 

import numpy as np
import matplotlib.pyplot as plt

aa = [3.35510803e+00, 4.41825516e-03, 1.27427389e-03, 3.61940735e+00,
 5.10212565e+00, 3.05248924e-03, 4.73520253e-03, 1.28228376e-03,
 1.55214601e+00, 5.10728430e+00]

fig = plt.figure(figsize=(6,2))
#plt.plot(aa, 'r*:')
plt.bar(range(1,11), aa)
plt.xticks(range(1,11), range(1,11))
plt.title('BF-VAE-1: Individual expected prior KL divs')
plt.ylabel(r'$E_{q(\alpha)}E_x[KL(q(z_j|x)||p(z_j))$')
plt.xlabel(r'$j$ in $z_j$')
plt.show()

fig.savefig('bfvae1_oval_indiv_kls.png')
fig.savefig('bfvae1_oval_indiv_kls.pdf')


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
