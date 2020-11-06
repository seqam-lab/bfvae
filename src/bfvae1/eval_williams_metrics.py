import os
import numpy as np
#import matplotlib.pyplot as plt

###############################################################################

#
# setups
#

# name of dataset
dataset ='oval_dsprites';  dset_dir = '../../new_vae/beta_vae/data'
#dataset ='3dfaces';  dset_dir = '../data'
#dataset ='edinburgh_teapots';  dset_dir = '../data'
#dataset ='celeba';  dset_dir = '../data'

# your latent dimension
n_c = 10
#n_c = 20

# location of latent vectors predicted by your model
latent_file = 'outputs/' + \
    'save_z_oval_dsprites_gamma_35.0_zDim_10' + \
    '_lrVAE_0.0001_lrD_0.0001_rseed_11_run_0.npz'
#latent_file = 'outputs/' + \
#    'save_z_3dfaces_etaS_5.0_etaH_5.0_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_10_run_0.npz'
#latent_file = 'outputs/' + \
#    'save_z_edinburgh_teapots_etaS_0.5_etaH_0.5_lamklMin_0.1_lamklMax_2.0' + \
#    '_gamma_10.0_zDim_10_run_0.npz'
#latent_file = 'outputs/' + \
#    'save_z_celeba_etaS_10.0_etaH_10.0_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_20_run_0.npz'
#latent_file = 'outputs/' + \
#    'save_z_celeba_etaS_1.0_etaH_1.0_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_20_run_0.npz'
#latent_file = 'outputs/' + \
#    'save_z_celeba_etaS_0.1_etaH_0.1_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_20_run_0.npz'

# where the metric evaluation results will be saved
record_file = 'records/' + \
    'williams_oval_dsprites_gamma_35.0_zDim_10' + \
    '_lrVAE_0.0001_lrD_0.0001_rseed_11_run_0.txt'
#record_file = 'records/' + \
#    'williams_3dfaces_etaS_5.0_etaH_5.0_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_10_run_0.txt'
#record_file = 'records/' + \
#    'williams_edinburgh_teapots_etaS_0.5_etaH_0.5_lamklMin_0.1_lamklMax_2.0' + \
#    '_gamma_10.0_zDim_10_run_0.txt'
#record_file = 'records/' + \
#    'williams_celeba_etaS_10.0_etaH_10.0_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_20_run_0.txt'
#record_file = 'records/' + \
#    'williams_celeba_etaS_1.0_etaH_1.0_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_20_run_0.txt'
#record_file = 'records/' + \
#    'williams_celeba_etaS_0.1_etaH_0.1_lamklMin_0.01_lamklMax_10.0' + \
#    '_gamma_10.0_zDim_20_run_0.txt'
    
#--#

seed = 123
rng = np.random.RandomState(seed)

###############################################################################

#
# helper functions
#

TINY = 1e-12

#-----------------------------------------------------------------------------#

def dump_to_record(record_file, prn_str):
    
    record = open(record_file, 'a')
    record.write('%s\n' % (prn_str,))
    record.close()
    
#-----------------------------------------------------------------------------#

def mkdir_p(path):
    os.makedirs(path)

#-----------------------------------------------------------------------------#
    
# split inputs and targets into sets: [train, dev, test]
def split_data(data, n_train, n_dev, n_test):
    
    train = data[:n_train]
    dev = data[n_train: n_train + n_dev]
    test = data[n_train + n_dev: n_train + n_dev + n_test]
    
    return [train, dev, test]

#-----------------------------------------------------------------------------#

def normalize( X, mean=None, stddev=None, useful_features=None, 
               remove_constant=True ):
    
    calc_mean, calc_stddev = False, False
    
    if mean is None:
        mean = np.mean(X, 0) # training set
        calc_mean = True
    
    if stddev is None:
        stddev = np.std(X, 0) # training set
        calc_stddev = True
        useful_features = np.nonzero(stddev)[0] 
            # inconstant features, ([0]=shape correction)
    
    if remove_constant and useful_features is not None:
        X = X[:, useful_features]
        if calc_mean:
            mean = mean[useful_features]
        if calc_stddev:
            stddev = stddev[useful_features]
    
    X_zm = X - mean    
    X_zm_unit = X_zm / (TINY + stddev)
    
    return X_zm_unit, mean, stddev, useful_features

#-----------------------------------------------------------------------------#

# normalize input and target datasets
def normalize_datasets(datasets):
    
    datasets[0], m, s, _ = normalize(datasets[0], remove_constant=False)
    datasets[1], _, _, _ = normalize(datasets[1], m, s, remove_constant=False)
    datasets[2], _, _, _ = normalize(datasets[2], m, s, remove_constant=False)

    return datasets

#-----------------------------------------------------------------------------#

def norm_entropy(p):
    
    '''p: probabilities '''

    n = p.shape[0]

    return - p.dot(np.log(p + TINY) / np.log(n + TINY))

#-----------------------------------------------------------------------------#
    
def entropic_scores(r):
    
    '''r: relative importances '''
    
    r = np.abs(r)
    ps = r / (TINY + np.sum(r, axis=0)) # 'probabilities'
    hs = [1-norm_entropy(p) for p in ps.T]
    
    return hs

#-----------------------------------------------------------------------------#
    
def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted 
        # (n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target 
        # (n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / (TINY + len(err))
    return err[0, 0]  # value not array

def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / (TINY + np.var(target))

def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / (TINY + np.std(target))

#-----------------------------------------------------------------------------#

def print_table_pretty(name, values, factor_label, model_names):

    headers = [factor_label + str(i) for i in range(len(values[0]))]
    headers[-1] = "Avg."
    headers = "\t" + "\t".join(headers)
    prn_str = "{0}:\n{1}".format(name, headers)
    print(prn_str)
    dump_to_record(record_file, prn_str)
    
    for i, values in enumerate(values):
        value = ""
        for v in values:
            value +=  "{0:.2f}".format(v) + "&\t"
        prn_str = "{0}\t{1}".format(model_names[i], value)
        print(prn_str)
        dump_to_record(record_file, prn_str)
    prn_str = ""  # newline
    print(prn_str)
    dump_to_record(record_file, prn_str)
        

###############################################################################

#
# prepare (load true factors and model's latent vectors)
#

#data_dir = 'data/' #'../wgan/data/' 
#codes_dir = os.path.join(data_dir, 'codes/')
#figs_dir = 'figs/'
#mkdir_p(figs_dir)

#model_names = ['PCA', 'VAE', '$\\beta$-VAE', 'InfoGAN']
#
#exp_names = [m.lower() for m in model_names]
#n_models = len(model_names)

## load inputs (model codes)
#m_codes = []
#for n in exp_names:
#    m_codes.append(np.load(os.path.join(codes_dir, n + '.npy')))

# load latent vectors and true factors
dd = np.load(latent_file)
m_codes, gts = dd['name1'], dd['name2']

n_samples = gts.shape[0]
n_z = gts.shape[1]

# split data into tr/va/te
tr_frac, va_frac, te_frac = 0.8, 0.1, 0.1
n_train, n_dev, n_test = \
    int(tr_frac*n_samples), int(va_frac*n_samples), int(te_frac*n_samples)
gts = split_data(gts, n_train, n_dev, n_test)
m_codes = split_data(m_codes, n_train, n_dev, n_test)    

# normalize data
gts = normalize_datasets(gts)
m_codes = normalize_datasets(m_codes)

    
###############################################################################

#
# fit regression model
#

def fit_visualise_quantify( regressor, params, err_fn, importances_attr, 
                            test_time=False, save_plot=False ):
    
    # lists to store scores
    m_disent_scores = []
    m_complete_scores = []
    
    # arrays to store errors (+1 for avg)
    train_errs = np.zeros([1,n_z+1])
    dev_errs   = np.zeros([1,n_z+1])
    test_errs  = np.zeros([1,n_z+1]) 
    
#    # init plot (Hinton diag)
#    fig, axs = plt.subplots(1,n_models, figsize=(12, 6), facecolor='w', edgecolor='k')
#    axs = axs.ravel()
    
    # init inputs
    X_train, X_dev, X_test = m_codes[0], m_codes[1], m_codes[2]
       
    # R_ij = relative importance of c_i in predicting z_j
    R = [] 
    
    # for each true factor
    for j in range(n_z):
        
        # init targets [shape=(n_samples, 1)]
        y_train = gts[0][:,j]
        y_dev = gts[1][:,j]
        y_test = gts[2][:,j] if test_time else None
        
        # fit model
        model = regressor(**params[j])
        model.fit(X_train, y_train)

        # predict
        y_train_pred = model.predict(X_train)
        y_dev_pred = model.predict(X_dev)
        y_test_pred = model.predict(X_test) if test_time else None
        
        # calculate errors
        train_errs[0,j] = err_fn(y_train_pred, y_train)
        dev_errs[0,j] = err_fn(y_dev_pred, y_dev)
        test_errs[0,j] = err_fn(y_test_pred, y_test) if test_time else None        
        
        # extract relative importance of each code var in predicting z_j
        r = getattr(model, importances_attr)[:, None] # [n_c, 1]
        R.append(np.abs(r))

    R = np.hstack(R) #columnwise, predictions of each z

    # disentanglement
    disent_scores = entropic_scores(R.T)
    c_rel_importance = np.sum(R,1) / (TINY + np.sum(R)) 
        # relative importance of each code variable
    disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
    disent_scores.append(disent_w_avg)
    m_disent_scores.append(disent_scores)

    # completeness
    complete_scores = entropic_scores(R)
    complete_avg = np.mean(complete_scores)
    complete_scores.append(complete_avg)
    m_complete_scores.append(complete_scores)

    # informativeness (append averages)
    train_errs[0,-1] = np.mean(train_errs[0,:-1])
    dev_errs[0,-1] = np.mean(dev_errs[0,:-1])
    test_errs[0,-1] = np.mean(test_errs[0,:-1]) if test_time else None
    
        
#        # visualise
#        hinton(R, '$\mathbf{z}$', '$\mathbf{c}$', ax=axs[i], fontsize=18)
#        axs[i].set_title('{0}'.format(model_names[i]), fontsize=20)
    
#    plt.rc('text', usetex=True)
#    if save_plot:
#        fig.tight_layout()
#        plt.savefig(os.path.join(figs_dir, "hint_{0}_{1}.pdf".format(regressor.__name__, n_c)))
#    else:
#        plt.show()
    
    prn_str = "<<<< " + regressor.__name__ + " >>>>\n"
    print(prn_str)
    dump_to_record(record_file, prn_str)

    print_table_pretty('Disentanglement', m_disent_scores, 'c', 'model')
    print_table_pretty('Completeness', m_complete_scores, 'z', 'model')

    print("Informativeness:")    
    print_table_pretty('Training Error', train_errs, 'z', 'model')
    print_table_pretty('Validation Error', dev_errs, 'z', 'model')    
    if test_time:
        print_table_pretty('Test Error', test_errs, 'z', 'model')

    
###############################################################################

#
# run it
#

#-----------------------------------------------------------------------------#
# LASSO

from sklearn.linear_model import Lasso

alpha = 0.02
if dataset == 'oval_dsprites':
    alpha = 0.2
if dataset == '3dfaces':
    alpha = 0.2
params = [{"alpha": alpha}]*n_z  # constant alpha for all models and targets

importances_attr = 'coef_' # weights
err_fn = nrmse # norm root mean sq. error
test_time = True
save_plot = False

fit_visualise_quantify( Lasso, params, err_fn, importances_attr, test_time, 
                        save_plot )


#-----------------------------------------------------------------------------#
# random forest

from sklearn.ensemble.forest import RandomForestRegressor

n_estimators = 10
all_best_depths = [5]*n_z
if dataset == 'oval_dsprites':
    all_best_depths = [12, 5, 12, 12]
if dataset == '3dfaces':
    all_best_depths = [3, 3, 3, 3]

# populate params dict with best_depths per model per target (z gt)
params = []
for z_max_depth in all_best_depths:
    params.append( { "n_estimators": n_estimators, "max_depth": z_max_depth, 
                     "random_state": rng} )

importances_attr = 'feature_importances_'
err_fn = nrmse # norm root mean sq. error
test_time = True
save_plot = False

fit_visualise_quantify( RandomForestRegressor, params, err_fn, 
                        importances_attr, test_time, save_plot )


