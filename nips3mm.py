"""
HCP: Semi-supervised network decomposition by low-rank logistic regression
"""
import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
os.environ["THEANO_FLAGS"] = "floatX=float32"
import theano
import theano.tensor as T
from matplotlib import pylab as plt
from nilearn.image import concat_imgs
import joblib
import time
import pandas as pd
from cogspaces.datasets import fetch_atlas_modl
from sklearn.model_selection import StratifiedShuffleSplit
from nilearn import datasets

print('Running THEANO on %s' % theano.config.device)
print(__doc__)

RES_NAME = 'nips3mm'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

##############################################################################
# load+preprocess data
##############################################################################

mask_img = 'goodvoxmask_hcp.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=None,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_fdata().sum()

print('Loading data...')

# @Gabighz - Modified to load locally available data
# downloaded from the HCP S500 Release Subjects (as part of WU-Minn HCP Data - 1200 Subjects),
# S900 Release Subjects
# Gambling Task fMRI Preprocessed and Cogspace resting-state dictionaries
task_img = 'GAMBLING_masked_1.nii.gz'
rest_img = 'components_128.nii.gz'
ARCHI_out_of_dataset = 'GAMBLING_masked_2.nii.gz'

# ARCHI task
# Modified by @Gabighz: previously was X_task, labels = joblib.load('preload_HT_3mm')
X_task = nifti_masker.fit_transform(task_img)

# @Gabighz
# Taken from Gambling_Stats.cvs
meta_file = pd.read_csv('GAMBLING_meta.csv', nrows=433)

labels = meta_file['Label']

AT_meta_file = pd.read_csv('GAMBLING_meta.csv', skiprows = (1, 433), nrows=348)
AT_labels = AT_meta_file['Label']

labels = [0 if label==-1 else label for label in labels]
AT_labels = [0 if label==-1 else label for label in AT_labels]

labels = np.int32(labels)
AT_labels = np.int32(AT_labels)

# contrasts are IN ORDER -> shuffle!
new_inds = np.arange(0, X_task.shape[0])
np.random.shuffle(new_inds)
X_task = X_task[new_inds]
labels = labels[new_inds]
# subs = subs[new_inds]

# Resting state data
# Modified by @Gabighz
X_rest = nifti_masker.transform(rest_img)
# X_rest = nifti_masker.transform('dump_rs_spca_s12_tmp')
rs_spca_data = nib.load(rest_img) # was: joblib.load('dump_rs_spca_s12_tmp')
rs_spca_niis = nib.Nifti1Image(rs_spca_data,
                               nifti_masker.mask_img_.affine)
#X_rest = nifti_masker.transform(rs_spca_niis)
del rs_spca_niis
del rs_spca_data

X_task = StandardScaler().fit_transform(X_task)
X_rest = StandardScaler().fit_transform(X_rest)

# ARCHI Task
# NOTE: Used to test out-of-dataset performance
# was AT_niis, AT_labels, AT_subs = joblib.load('preload_AT_3mm')
AT_niis = nib.load(ARCHI_out_of_dataset)
AT_X = nifti_masker.transform(AT_niis)
AT_X = StandardScaler().fit_transform(AT_X)
print('done :)')


#bg_img = nib.load(mask_img)
#bg = bg_img.get_data()
#act = bg.copy()
#act[act < 6000] = 0

#plt.imshow(bg[..., 10].T, origin='lower', interpolation = 'nearest', cmap = 'gray')
#masked_act = np.ma.masked_equal(act, 0.)
#plt.imshow(masked_act[..., 10].T, origin='lower', interpolation='nearest', cmap='hot')
#plt.axis('off')
#plt.show()

##############################################################################
# define computation graph
##############################################################################

class SSEncoder(BaseEstimator):
    def __init__(self, n_hidden, gain1, learning_rate, max_epochs=100,
                 l1=0.1, l2=0.1, lambda_param=.5):
        """
        Parameters
        ----------
        lambda : float
            Mediates between AE and LR. lambda==1 equates with LR only.
        """
        self.n_hidden = n_hidden
        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = np.float32(learning_rate)
        self.penalty_l1 = np.float32(l1)
        self.penalty_l2 = np.float32(l2)
        self.lambda_param = np.float32(lambda_param)

    # def rectify(X):
    #     return T.maximum(0., X)

    from theano.tensor.shared_randomstreams import RandomStreams

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        gradients = T.grad(cost=cost, wrt=params)
        updates = []
        for param, gradient in zip(params, gradients):
            acc = theano.shared(param.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * gradient ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            gradient = gradient / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((param, param - lr * gradient))
        return updates
        
    def get_param_pool(self):
        cur_params = (
            self.V1s, self.bV0, self.bV1,
            self.W0s, self.W1s, self.bW0s, self.bW1s
        )
        return cur_params
        
    def test_performance_in_other_dataset(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedShuffleSplit

        compr_matrix = self.W0s.get_value().T  # currently best compression
        AT_X_compr = np.dot(compr_matrix, AT_X.T).T
        clf = LogisticRegression(penalty='l2') #Modified: was penalty l1
        # Modified
        #folder = StratifiedShuffleSplit(y=AT_labels, n_iter=5, test_size=0.2,
        #                                random_state=42)

        folder = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=42)

        acc_list = []
        prfs_list = []
        for (train_inds, test_inds) in folder.split(AT_X_compr, AT_labels):
            clf.fit(AT_X_compr[train_inds, :], AT_labels[train_inds])
            pred_y = clf.predict(AT_X_compr[test_inds, :])

            acc = (pred_y == AT_labels[test_inds]).mean()
            prfs_list.append(precision_recall_fscore_support(
                             AT_labels[test_inds], pred_y))

            acc_list.append(acc)

        compr_mean_acc = np.mean(acc_list)
        prfs = np.asarray(prfs_list).mean(axis=0)
        return compr_mean_acc, prfs

    def fit(self, X_rest, X_task, y):
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        n_input = X_rest.shape[1]  # sklearn-like structure
        n_output = n_input
        rng = np.random.RandomState(42)
        self.input_taskdata = T.matrix(dtype='float32', name='input_taskdata')
        self.input_restdata = T.matrix(dtype='float32', name='input_restdata')
        self.params_from_last_iters = []

        index = T.iscalar(name='index')
        
        # prepare data for theano computation
        if not DEBUG_FLAG:
            X_train_s = theano.shared(
                value=np.float32(X_task), name='X_train_s')
            y_train_s = theano.shared(
                value=np.int32(y), name='y_train_s')
            lr_train_samples = len(X_task)
        else:
            # Modified
            # folder = StratifiedShuffleSplit(y, n_splits=1, test_size=0.20)
            folder = StratifiedShuffleSplit(n_splits=1, test_size=0.20)
            #folder.split(np.zeros(len(y)), y)
            #new_trains, inds_val = iter(folder).next()
            for train_index, test_index in folder.split(X_task, y):
                X_train, X_val = X_task[train_index], X_task[test_index]
                y_train, y_val = y[train_index], y[test_index]

            X_train_s = theano.shared(value=np.float32(X_train),
                                      name='X_train_s', borrow=False)
            y_train_s = theano.shared(value=np.int32(y_train),
                                      name='y_train_s', borrow=False)
            # X_val_s = theano.shared(value=np.float32(X_val),
            #                         name='X_train_s', borrow=False)
            # y_val_s = theano.shared(value=np.int32(y_val),
            #                         name='y_cal_s', borrow=False)
            lr_train_samples = len(X_train)
            self.dbg_epochs_ = list()
            self.dbg_acc_train_ = list()
            self.dbg_acc_val_ = list()
            self.dbg_ae_cost_ = list()
            self.dbg_lr_cost_ = list()
            self.dbg_ae_nonimprovesteps = list()
            self.dbg_acc_other_ds_ = list()
            self.dbg_combined_cost_ = list()
            self.dbg_prfs_ = list()
            self.dbg_prfs_other_ds_ = list()
        X_rest_s = theano.shared(value=np.float32(X_rest), name='X_rest_s')
        ae_train_samples = len(X_rest)

        # V -> supervised / logistic regression
        # W -> unsupervised / auto-encoder

        # computational graph: auto-encoder
        W0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1

        self.W0s = theano.shared(W0_vals)
        self.W1s = self.W0s.T  # tied
        bW0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bW0s = theano.shared(value=bW0_vals, name='bW0')
        bW1_vals = np.zeros(n_output).astype(np.float32)
        self.bW1s = theano.shared(value=bW1_vals, name='bW1')

        givens_ae = {
            self.input_restdata: X_rest_s[
                index * self.batch_size:(index + 1) * self.batch_size]
        }

        encoding = (self.input_restdata.dot(self.W0s) + self.bW0s).dot(self.W1s) + self.bW1s

        self.ae_loss = T.sum((self.input_restdata - encoding) ** 2, axis=1)

        self.ae_cost = (
            T.mean(self.ae_loss) / n_input
        )

        # params1 = [self.W0s, self.bW0s, self.bW1s]
        # gparams1 = [T.grad(cost=self.ae_cost, wrt=param1) for param1 in params1]
        # 
        # lr = self.learning_rate
        # updates = self.RMSprop(cost=self.ae_cost, params=params1,
        #                        lr=self.learning_rate)

        # f_train_ae = theano.function(
        #     [index],
        #     [self.ae_cost],
        #     givens=givens_ae,
        #     updates=updates)

        # computation graph: logistic regression
        clf_n_output = 18  # number of labels
        my_y = T.ivector(name='y')

        bV0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bV0 = theano.shared(value=bV0_vals, name='bV0')
        bV1_vals = np.zeros(clf_n_output).astype(np.float32)
        self.bV1 = theano.shared(value=bV1_vals, name='bV1')
        
        # V0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1
        V1_vals = rng.randn(self.n_hidden, clf_n_output).astype(np.float32) * self.gain1
        # self.V0s = theano.shared(V0_vals)
        self.V1s = theano.shared(V1_vals)

        self.p_y_given_x = T.nnet.softmax(
            # T.dot(T.dot(self.input_taskdata, self.V0s) + self.bV0, self.V1s) + self.bV1
            T.dot(T.dot(self.input_taskdata, self.W0s) + self.bV0, self.V1s) + self.bV1
        )
        self.lr_cost = -T.mean(T.log(self.p_y_given_x)[T.arange(my_y.shape[0]), my_y])
        self.lr_cost = (
            self.lr_cost +
            T.mean(abs(self.W0s)) * self.penalty_l1 +
            # T.mean(abs(self.V0s)) * self.penalty_l1 +
            T.mean(abs(self.bV0)) * self.penalty_l1 +
            T.mean(abs(self.V1s)) * self.penalty_l1 +
            T.mean(abs(self.bV1)) * self.penalty_l1 +

            T.mean((self.W0s ** np.float32(2))) * self.penalty_l2 +
            # T.mean((self.V0s ** 2)) * self.penalty_l2 +
            T.mean((self.bV0 ** np.float32(2))) * self.penalty_l2 +
            T.mean((self.V1s ** np.float32(2))) * self.penalty_l2 +
            T.mean((self.bV1 ** np.float32(2))) * self.penalty_l2
        )
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        givens_lr = {
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }

        # params2 = [self.V0s, self.bV0, self.V1s, self.bV1]
        # params2 = [self.W0s, self.bV0, self.V1s, self.bV1]
        # updates2 = self.RMSprop(cost=self.lr_cost, params=params2,
        #                         lr=self.learning_rate)

        # f_train_lr = theano.function(
        #     [index],
        #     [self.lr_cost],
        #     givens=givens_lr,
        #     updates=updates2)

        # combined loss for AE and LR
        combined_params = [self.W0s, self.bW0s, self.bW1s,
                        #    self.V0s, self.V1s, self.bV0, self.bV1]
                           self.V1s, self.bV0, self.bV1]
        self.combined_cost = (
            (np.float32(1) - self.lambda_param) * self.ae_cost +
            self.lambda_param * self.lr_cost
        )
        np.combined_updates = self.RMSprop(
            cost=self.combined_cost,
            params=combined_params,
            lr=self.learning_rate)
        givens_combined = {
            self.input_restdata: X_rest_s[index * self.batch_size:(index + 1) * self.batch_size],
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }
        f_train_combined = theano.function(
            [index],
            [self.combined_cost, self.ae_cost, self.lr_cost],
            givens=givens_combined,
            updates=np.combined_updates, allow_input_downcast=True)

        # optimization loop
        start_time = time.time()
        ae_last_cost = np.inf
        # Modified
        # ae_cur_cost, lr_cur_cost, and combined_cost used to be referenced before assignment
        # in lines 'if ae_last_cost - ae_cur_cost < 0.1:',
        # 'lr_last_cost = lr_cur_cost', and 'self.dbg_combined_cost_.append(combined_cost)'
        ae_cur_cost = np.inf
        lr_cur_cost = np.inf
        lr_last_cost = np.inf
        combined_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))

            # AE
            # if i_epoch % 2 == 0:  # every second time
            #if False:
                # auto-encoder
            ae_n_batches = ae_train_samples // self.batch_size
            lr_n_batches = lr_train_samples // self.batch_size
            # for i in range(lr_n_batches):
            # for i in range(max(ae_n_batches, lr_n_batches)):
                # if i < ae_n_batches:
                #     ae_cur_cost = float(f_train_ae(i)[0])
                # ae_cur_cost = 0
                # if i < lr_n_batches:
                #     lr_cur_cost = float(f_train_lr(i)[0])
            # for i in range(lr_n_batches):
            for i in range(min(ae_n_batches, lr_n_batches)):
                # lr_cur_cost = f_train_lr(i)[0]
                # ae_cur_cost = lr_cur_cost
                combined_cost, ae_cur_cost, lr_cur_cost = f_train_combined(i)

            # evaluate epoch cost
            if ae_last_cost - ae_cur_cost < 0.1:
                no_improve_steps += 1
            else:
                ae_last_cost = ae_cur_cost
                no_improve_steps = 0

            # logistic
            lr_last_cost = lr_cur_cost
            acc_train = self.score(X_train, y_train)
            acc_val, prfs_val = self.score(X_val, y_val, return_prfs=True)

            print('E:%i, ae_cost:%.4f, lr_cost:%.4f, train_score:%.2f, vald_score:%.2f, ae_badsteps:%i' % (
                i_epoch + 1, ae_cur_cost, lr_cur_cost, acc_train, acc_val, no_improve_steps))

            if (i_epoch % 10 == 0):
                self.dbg_ae_cost_.append(ae_cur_cost)
                self.dbg_lr_cost_.append(lr_cur_cost)
                self.dbg_combined_cost_.append(combined_cost)

                self.dbg_epochs_.append(i_epoch + 1)
                self.dbg_ae_nonimprovesteps.append(no_improve_steps)
                self.dbg_acc_train_.append(acc_train)
                self.dbg_acc_val_.append(acc_val)
                self.dbg_prfs_.append(prfs_val)

                # test out-of-dataset performance
                od_acc, prfs_other = self.test_performance_in_other_dataset()
                self.dbg_acc_other_ds_.append(od_acc)
                self.dbg_prfs_other_ds_.append(prfs_other)
                print('out-of-dataset acc: %.2f' % od_acc)
                
            # save paramters from last 100 iterations
            if i_epoch > (self.max_epochs - 100):
                print('Param pool!')
                param_pool = self.get_param_pool()
                self.params_from_last_iters.append(param_pool)

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def predict(self, X):
        X_test_s = theano.shared(value=np.float32(X), name='X_test_s', borrow=True)

        givens_te = {
            self.input_taskdata: X_test_s
        }

        f_test = theano.function(
            [],
            [self.y_pred],
            givens=givens_te)
        predictions = f_test()
        del X_test_s
        del givens_te
        return predictions[0]

    def score(self, X, y, return_prfs=False):
        pred_y = self.predict(X)
        acc = np.mean(pred_y == y)
        # @Gabighz
        # Added zero_division parameter
        prfs = precision_recall_fscore_support(pred_y, y, zero_division=1)
        if return_prfs:
            return acc, prfs
        else:
            return acc


##############################################################################
# plot figures
##############################################################################

def dump_comps(masker, compressor, components, threshold=2, fwhm=None,
               perc=None):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map
    from nilearn.image import smooth_img
    from scipy.stats import scoreatpercentile

    if isinstance(compressor, str):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')
        
        comp_z = zscore(comp)
        
        if perc is not None:
            cur_thresh = scoreatpercentile(np.abs(comp_z), per=perc)
            path_mask += '_perc%i' % perc
            print('Applying percentile %.2f (threshold: %.2f)' % (perc, cur_thresh))
        else:
            cur_thresh = threshold
            path_mask += '_thr%.2f' % cur_thresh
            print('Applying threshold: %.2f' % cur_thresh)

        nii_z = masker.inverse_transform(comp_z)
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img=localizer_anat_file, threshold=cur_thresh,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')
                      
        # optional: do smoothing
        if fwhm is not None:
            nii_z_fwhm = smooth_img(nii_z, fwhm=fwhm)
            plot_stat_map(nii_z_fwhm, bg_img=localizer_anat_file, threshold=cur_thresh,
                          cut_coords=(0, -2, 0), draw_cross=False,
                          output_file=path_mask +
                          ('zmap_%imm.png' % fwhm))
# n_comps = [5, 20, 50, 100]
n_comps = [20]
for n_comp in n_comps:
    # for lambda_param in [0]:
    for lambda_param in [0.50]:
        l1 = 0.1
        l2 = 0.1
        my_title = r'Low-rank LR + AE (combined loss, shared decomp): n_comp=%i L1=%.1f L2=%.1f lambda=%.2f res=3mm spca20RS' % (
            n_comp, l1, l2, lambda_param
        )
        print(my_title)
        estimator = SSEncoder(
            n_hidden=n_comp,
            gain1=0.004,  # empirically determined by CV
            learning_rate = np.float32(0.00001),  # empirically determined by CV,
            max_epochs=500, l1=l1, l2=l2, lambda_param=lambda_param)
        
        estimator.fit(X_rest, X_task, labels)

        fname = my_title.replace(' ', '_').replace('+', '').replace(':', '').replace('__', '_').replace('%', '')
        cur_path = op.join(WRITE_DIR, fname)
        joblib.dump(estimator, cur_path)
        # estimator = joblib.load(cur_path)
        # plt.savefig(cur_path + '_SUMMARY.png', dpi=200)
        
        # dump data also as numpy array
        np.save(cur_path + 'dbg_epochs_', np.array(estimator.dbg_epochs_))
        np.save(cur_path + 'dbg_acc_train_', np.array(estimator.dbg_acc_train_))
        np.save(cur_path + 'dbg_acc_val_', np.array(estimator.dbg_acc_val_))
        np.save(cur_path + 'dbg_ae_cost_', np.array(estimator.dbg_ae_cost_))
        np.save(cur_path + 'dbg_lr_cost_', np.array(estimator.dbg_lr_cost_))
        np.save(cur_path + 'dbg_ae_nonimprovesteps', np.array(estimator.dbg_ae_nonimprovesteps))
        np.save(cur_path + 'dbg_acc_other_ds_', np.array(estimator.dbg_acc_other_ds_))
        np.save(cur_path + 'dbg_combined_cost_', np.array(estimator.dbg_combined_cost_))
        np.save(cur_path + 'dbg_prfs_', np.array(estimator.dbg_prfs_))
        np.save(cur_path + 'dbg_prfs_other_ds_', np.array(estimator.dbg_prfs_other_ds_))

        W0_mat = estimator.W0s.get_value().T
        np.save(cur_path + 'W0comps', W0_mat)
        
        V1_mat = estimator.V1s.get_value().T
        np.save(cur_path + 'V1comps', V1_mat)
        # dump_comps(nifti_masker, fname, comps, threshold=0.5)

# Modified by @Gabighz
# Commented out the line below since it was throwing a
# variable undefined error:
# STOP_CALCULATION

# equally scaled plots
import re
pkgs = glob.glob(RES_NAME + '/*dbg_epochs*.npy')
dbg_epochs_ = np.load(pkgs[0])
dbg_epochs_ = np.load(pkgs[0])

data = {
    'training accuracy': '/*dbg_acc_train*.npy',
    'accuracy val': '/*dbg_acc_val_*.npy',
    'accuracy other ds': '/*dbg_acc_other_ds_*.npy',
    'loss ae': '/*dbg_ae_cost_*.npy',
    'loss lr': '/*dbg_lr_cost_*.npy',
    'loss combined': '/*dbg_combined_cost_*.npy'
}
n_comps = [20]

path_vanilla = 'nips3mm_vanilla'

for key, value in data.items():
    pkgs = glob.glob(RES_NAME + value)
    for n_comp in n_comps:
        plt.figure()
        for package in pkgs:
            lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
            # n_hidden = int(re.search('comp=(?P<comp>.{1,2,3})_', package).group('comp'))
            n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
            if n_comp != n_hidden:
                continue
            
            dbg_acc_train_ = np.load(package)
            
            cur_label = 'n_comp=%i' % n_hidden
            cur_label += '/'
            cur_label += 'lambda=%.2f' % lambda_param
            cur_label += '/'
            if not '_AE' in package:
                cur_label += 'LR only!'
            elif 'subRS' in package:
                cur_label += 'RSnormal'
            elif 'spca20RS' in package:
                cur_label += 'RSspca20'
            elif 'pca20RS' in package:
                cur_label += 'RSpca20'
            cur_label += '/'
            cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'
            cur_label += '' if '_AE' in package else '/LR only!'
            plt.plot(
                dbg_epochs_,
                dbg_acc_train_,
                label=cur_label)
        if key == 'training accuracy' or key == 'accuracy val':
            van_pkgs = glob.glob(path_vanilla + value)

            vanilla_values = np.load(van_pkgs[0])
            plt.plot(
               dbg_epochs_,
               vanilla_values,
               label='LR')
        plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
        plt.legend(loc='lower right', fontsize=9)
        plt.yticks(np.linspace(0., 1., 11))
        plt.ylabel(key)
        plt.xlabel('epochs')
        plt.ylim(0., 1.05)
        plt.grid(True)
        plt.show()
        plt.savefig(op.join(WRITE_DIR,
                    key.replace(' ', '_') + '_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_acc_val_*.npy')
for n_comp in n_comps:  # 
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        # n_hidden = int(re.search('comp=(?P<comp>.{1,2,3})_', package).group('comp'))
        n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_acc_val_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        plt.plot(
            dbg_epochs_,
            dbg_acc_val_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('validation set accuracy')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'accuracy_val_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_acc_other_ds_*.npy')
for n_comp in n_comps:  # 
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_acc_other_ds_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        plt.plot(
            dbg_epochs_,
            dbg_acc_other_ds_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('ARCHI dataset accuracy')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'accuracy_archi_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_ae_cost_*.npy')
for n_comp in n_comps:  # AE
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_ae_cost_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_ae_cost_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('AE loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'loss_ae_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_lr_cost_*.npy')  # LR cost
for n_comp in n_comps:  # AE
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_lr_cost_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_lr_cost_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('LR loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'loss_lr_%icomps.png' % n_comp))

pkgs = glob.glob(RES_NAME + '/*dbg_combined_cost_*.npy')  # combined loss
for n_comp in n_comps:  # AE
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
        if n_comp != n_hidden:
            continue
        
        dbg_combined_cost_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'
        plt.plot(
            dbg_epochs_,
            dbg_combined_cost_,
            label=cur_label)
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('combined loss')
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'loss_combined_%icomps.png' % n_comp))

# precision / recall / f1
target_lambda = 0.5

pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', package).group('comp'))
        if n_comp != n_hidden:
            continue
        
        dbg_prfs_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        for i in np.arange(18):
             plt.plot(
                 dbg_epochs_,
                 np.array(dbg_prfs_)[:, 0, i],
                 label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('in-dataset precisions')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'prec_inds_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# in-dataset recall at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', package).group('comp'))
        if n_comp != n_hidden:
            continue
        
        dbg_prfs_ = np.load(package)
        
        dbg_prfs_ = np.load(package)
        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_)[:, 1, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('in-dataset recall')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'rec_inds_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# in-dataset f1 at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', package).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_ = np.load(package)
            
        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_)[:, 2, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('in-dataset f1 score')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'f1_inds_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# out-of-dataset precision at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_other_ds_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', package).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_other_ds_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_other_ds_)[:, 0, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('out-of-dataset precisions')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'prec_oods_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# out-of-dataset recall at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_other_ds_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', package).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_other_ds_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_other_ds_)[:, 1, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('out-of-dataset recall')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'rec_oods_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# out-of-dataset f1 at lambda=0.5
pkgs = glob.glob(RES_NAME + '/*lambda=%.2f*dbg_prfs_other_ds_.npy' % target_lambda)
for n_comp in n_comps:
    plt.figure()
    for package in pkgs:
        lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
        n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', package).group('comp'))
        if n_comp != n_hidden:
            continue
            
        dbg_prfs_other_ds_ = np.load(package)

        cur_label = 'n_comp=%i' % n_hidden
        cur_label += '/'
        cur_label += 'lambda=%.2f' % lambda_param
        cur_label += '/'
        if not '_AE' in package:
            cur_label += 'LR only!'
        elif 'subRS' in package:
            cur_label += 'RSnormal'
        elif 'pca20RS' in package:
            cur_label += 'RSpca20'
        cur_label += '/'
        cur_label += 'separate decomp.' if 'decomp_separate' in package else 'joint decomp.'

        for i in np.arange(18):
            plt.plot(
                dbg_epochs_,
                np.array(dbg_prfs_other_ds_)[:, 2, i],
                label='task %i' % (i + 1))
    plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=%.2f' %
              target_lambda)
    plt.legend(loc='lower right', fontsize=9)
    # plt.yticks(np.linspace(0., 1., 11))
    plt.ylabel('out-of-dataset f1 score')
    plt.ylim(0., 1.05)
    plt.xlabel('epochs')
    plt.grid(True)
    plt.show()
    plt.savefig(op.join(WRITE_DIR, 'f1_oods_lambda=%0.2f_%icomps.png' %
                (target_lambda, n_comp)))

# print network components (1st layer)
from nilearn.image import smooth_img
n_comp = 20
lmbd = 0.25
pkgs = glob.glob(RES_NAME + '/*W0comps.npy')
for package in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
    if n_comp != n_hidden or lambda_param != lmbd:
        continue
        
    new_fname = 'comps_n=%i_lambda=%.2f_th0.0' % (n_hidden, lambda_param)
    comps = np.load(package)
    dump_comps(nifti_masker, new_fname, comps, threshold=0.0)

# print class weights
pkgs = glob.glob(RES_NAME + '/*W0comps.npy')
n_comp = 20
lmbd = 0.5
for package in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
    if n_comp != n_hidden or lambda_param != lmbd:
        continue
    print(package)
    
    q = package.replace('W0', 'V1')
    comps = np.dot(np.load(q), np.load(package))
        
    new_fname = 'comps_n=%i_lambda=%.2f' % (n_hidden, lambda_param)
    dump_comps(nifti_masker, new_fname, comps, threshold=0.0, fwhm=None,
               perc=75)
    

# print LR decision matrix (2nd layer)
n_comp = 20
lmbd = 0.5
pkgs = glob.glob(RES_NAME + '/*V1comps.npy')
for package in pkgs:
    lambda_param = np.float(re.search('lambda=(.{4})', package).group(1))
    n_hidden = int(re.search('comp=(.{1,3})_', package).group(1))
    if n_comp != n_hidden or lambda_param != lmbd:
        continue
    print(package)
    
    cur_mat = np.load(package)

    # @Gabighz
    # NOTE: what is fs?
    if n_comp == 20:
        fs = (8, 6)
    elif n_comp == 100:
        fs = (12, 8)
        

    plt.figure(figsize=fs)
    masked_data = np.ma.masked_where(cur_mat != 0., cur_mat)
    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.gray_r)
    masked_data = np.ma.masked_where(cur_mat == 0., cur_mat)
    plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.RdBu_r)
    plt.show()

    # plt.xticks(range(n_comp)[::5], (np.arange(n_comp) + 1)[::5])
    # plt.xlabel('hidden components')
    # plt.yticks(range(18), np.arange(18) + 1)
    # plt.ylabel('tasks')
    # plt.title('Linear combinations of component per task')
    # plt.colorbar()
    
    new_fname = 'comps_n=%i_lambda=%.2f_V1_net2task.png' % (n_hidden, lambda_param)
    plt.savefig(op.join(WRITE_DIR, new_fname))

# out-of-dataset f1 score summary plots
for n_comp in [20, 50, 100]:
    f1_mean_per_lambda = list()
    f1_std_per_lambda = list()
    lambs = [0.25, 0.5, 0.75, 1.0]
    for target_lambda in lambs:
        pkgs = glob.glob(RES_NAME + '/*n_comp=%i*lambda=%.2f*dbg_prfs_other_ds_.npy' %
            (n_comp, target_lambda))
        print(pkgs)

        dbg_prfs_other_ds_ = np.load(pkgs[0])
        cur_mean = np.mean(dbg_prfs_other_ds_[-1, 2, :])
        f1_mean_per_lambda.append(cur_mean)
        cur_std = np.std(dbg_prfs_other_ds_[-1, 2, :])
        f1_std_per_lambda.append(cur_std)
        print('F1 means: %.2f +/- %.2f (SD)' % (cur_mean, cur_std))

    f1_mean_per_lambda = np.array(f1_mean_per_lambda)
    f1_std_per_lambda = np.array(f1_std_per_lambda)

    plt.figure()
    ind = np.arange(4)
    width = 1.
    colors = [#(7., 116., 242.), #(7., 176., 242.)
        #(7., 136., 217.), (7., 40., 164.), (1., 4., 64.)]
        (7., 176., 242.), (7., 136., 217.), (7., 40., 164.), (1., 4., 64.)]
    my_colors = [(x/256, y/256, z/256) for x, y, z in colors]

    plt.bar(ind, f1_mean_per_lambda, yerr=f1_std_per_lambda,
            width=width, color=my_colors)
    plt.ylabel('mean F1 score (+/- SD)')
    plt.title('out-of-dataset performance\n'
              '%i components' % n_comp)
    tick_strs = [u'low-rank $\lambda=%.2f$' % val for val in lambs]
    plt.xticks(ind + width / 2., tick_strs, rotation=320)
    plt.ylim(.5, 1.0)
    plt.grid(True)
    plt.yticks(np.linspace(0.5, 1., 11), np.linspace(0.5, 1., 11))
    plt.tight_layout()
    out_path2 = op.join(WRITE_DIR, 'f1_bars_comp=%i.png' % n_comp)
    plt.savefig(out_path2)
