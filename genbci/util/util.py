# coding=utf-8
from torch.autograd import Variable
from torch.nn import Module
import numpy as np
import scipy.io as sio

def get_balanced_batches(
    n_trials, rng, shuffle, n_batches=None, batch_size=None
):
    """Create indices for batches balanced in size 
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).
    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional
    Returns
    -------
    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches
def get_data(subject,training,PATH):
    '''	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets
    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
                if False, load testing data
    
    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
            class_return 	numpy matrix 	size = NO_valid_trial
    '''
    NO_channels = 22
    NO_tests = 6*48 	
    Window_Length = 7*250 

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests,NO_channels,Window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2=[a_data1[0,0]]
        a_data3=a_data2[0]
        a_X 		= a_data3[0]
        a_trial 	= a_data3[1]
        a_y 		= a_data3[2]
        a_fs 		= a_data3[3]
        a_classes 	= a_data3[4]
        a_artifacts = a_data3[5]
        a_gender 	= a_data3[6]
        a_age 		= a_data3[7]
        for trial in range(0,a_trial.size):
            if(a_artifacts[trial]==0):
                data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial +=1


    return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

def cuda_check(module_list):
    """
    Checks if any module or variable in a list has cuda() true and if so
    moves complete list to cuda

    Parameters
    ----------
    module_list : list
        List of modules/variables

    Returns
    -------
    module_list_new : list
        Modules from module_list all moved to the same device
    """
    cuda = False
    for mod in module_list:
        if isinstance(mod,Variable): cuda = mod.is_cuda
        elif isinstance(mod,Module): cuda = next(mod.parameters()).is_cuda

        if cuda:
            break
    if not cuda:
        return module_list

    module_list_new = []
    for mod in module_list:
        module_list_new.append(mod.cuda())
    return module_list_new


def change_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('MultiConv') != -1:
        for conv in m.convs:
            conv.weight.data.normal_(0.0, 1.)
            if conv.bias is not None:
                conv.bias.data.fill_(0.)
    elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.) # From progressive GAN paper
        if m.bias is not None:
            m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.)
