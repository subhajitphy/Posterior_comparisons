import tensiometer
from tensiometer import mcmc_tension
import numpy as np
import numpy.linalg as nl
import scipy, math
from tensiometer import utilities
mcmc_tension.n_threads = 1
# create the distribution of the parameter differences:
import warnings
from tensorflow.keras.callbacks import ReduceLROnPlateau

def Diff_chain_shift(A1,A2,MLinfo=None,method=None):
    
    input_arr=[A1,A2]

    
    diff_chain = mcmc_tension.parameter_diff_chain( A1, A2)
    
    if len(A1.getMeans())<=2 and method==None:
        
        shift_probability, shift_lower, shift_upper = mcmc_tension.kde_parameter_shift_2D_fft(diff_chain, feedback=0)
        nsig=utilities.from_confidence_to_sigma(shift_probability)
        sigl=utilities.from_confidence_to_sigma(shift_lower)
        sigh=utilities.from_confidence_to_sigma(shift_upper)
        #nsig=utilities.from_confidence_to_sigma(shift_probability)
        if (shift_probability!=1):
            return diff_chain, nsig, sigl, sigh
        else:
            sig_n, chi_2, D_B, chi2_probability=tension_chi2_approach(A1,A2)
            warnings.warn('Tension is higher!! chi squared estimators will be used to compute the tension.')
            return diff_chain, sig_n, chi_2, D_B
        

            
    if method=='ML' or len(A1.getMeans())>2:

        if(MLinfo is None):
            batch_size = 8192
            epochs = 50
            steps_per_epoch = 64
            MLinfo=[batch_size,epochs,steps_per_epoch]
        diff_flow_callback = tensiometer.mcmc_tension.DiffFlowCallback(diff_chain, feedback=1, learning_rate=0.01)

        callbacks = [ReduceLROnPlateau()]

        diff_flow_callback.train(batch_size=MLinfo[0], epochs=MLinfo[1], steps_per_epoch=MLinfo[2], callbacks=callbacks)


        exact_shift_P_1, exact_shift_low_1, exact_shift_hi_1 = diff_flow_callback.estimate_shift()
        nsig=utilities.from_confidence_to_sigma(exact_shift_P_1)
        sigl=utilities.from_confidence_to_sigma(exact_shift_low_1)
        sigh=utilities.from_confidence_to_sigma(exact_shift_hi_1)
        
        
        if (exact_shift_P_1!=1):
            return diff_chain, nsig, sigl, sigh
        

        else:
            sig_n, chi_2, D_B, chi2_probability=tension_chi2_approach(A1,A2)
            warnings.warn('Tension is higher than 4 sigma!! chi squared estimators will be used to compute the tension.')
            return diff_chain, sig_n, chi_2, D_B
      
        
        
        
        
    

def tension_chi2_approach(A1,A2):
    
    mean_1=A1.getMeans()
    mean_2=A2.getMeans()
    cov_1=A1.cov()
    cov_2=A2.cov()
    diff_mean  = mean_1 -mean_2
    diff_cov   = cov_1 + cov_2
    dof=len(mean_1)
    icov=nl.inv(diff_cov)
    icov_2=nl.inv(diff_cov/2)
    pp=np.dot(diff_mean,np.dot(icov_2,diff_mean))
    
    D_B=(1/8*pp+1/2*np.log(nl.det(diff_cov/2)/np.sqrt(nl.det(cov_1)*nl.det(cov_2))))
    
    chi_2=np.dot(diff_mean,np.dot(icov,diff_mean))
    sig_n=utilities.from_chi2_to_sigma(chi_2,dof)
    
    chi2_probability = scipy.stats.chi2.cdf(chi_2,dof)
    
    
    return sig_n, chi_2, D_B, chi2_probability
