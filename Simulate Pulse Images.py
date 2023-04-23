
#Code that creates a simulated pulse in frequency channels
#-----------------------------------------------------------------------------

#import the important libraries
import numpy as np
from skimage.measure import block_reduce
from scipy.signal import fftconvolve


#Define important functions used on the code

def DM_delay(dm,frequencies):
    '''
    Function that calculates the time delay at a given DM calculated using the standard formula
    [dm] = int
    [frequencies] = array with frequencies ranges
    '''
    frequencies = frequencies/1000
    
    t_delays = 4.15*dm*(   (frequencies**-2) -  (frequencies[0]**-2)   )
       
        
    return t_delays/1000

def DM_max(time_observation,bot_freq,bw):
    
    dm_max=time_observation*1000/4.15*(1/(bot_freq/1000)**2 - 1/((bot_freq+bw)/1000)**2)**-1
    
    return dm_max

def t_burst_min(DM,freq):
    
    t_burst_min = -min(DM_delay(DM, freq))
    
    return t_burst_min

def flux_freq(initial_amplitude,initial_frequency,frequencies,alpha):
    
    flux_freq = initial_amplitude*(frequencies/initial_frequency)**alpha
    
    return flux_freq


def pulse_broadening(freqs, DM, pulse, obs_t,idx):
    
    tau_o = 10.0**idx/1000
        
    tau = tau_o*(freqs[-1]/freqs)**4
    
    tau = np.reshape(tau,[1,len(tau)])
        
    obs_t = np.reshape(obs_t, [20096,1])
    
    kernel = np.exp(-obs_t/tau)
    
    kernel /= kernel.sum(axis=0)
    
    conv_signal = fftconvolve(pulse,kernel, mode='full',axes = 0)
    
    conv_signal = conv_signal[:tstamp.shape[0],:nchannels]

    return conv_signal


#------------------------------------------------------------------------------
## This script generates a text file containing fake data from an FRB


#------------------------------------------------------------------------------
## Observing setup


## Input parameters for the observing setup
nchannels = 512
bottom_freq = 1200.
bandwidth = 400.
dt_sampling = 5e-5
obs_length = 1.

#Directory where to store file

DATA_DIR = ''  #directory where you want to store your pulse imahes
randID = np.random.uniform(0,1000)
randID = str(int(randID))
name_file = DATA_DIR+'pulse_properties'+randID

header = ['DM','duration of burst','amplitude of burst','latest time of arrival','index alpha','log tau_o']

## Derived quantities
chanwidth = bandwidth / nchannels
freqs = np.arange(bottom_freq, bottom_freq+bandwidth, chanwidth)
tstamp = np.linspace(0, obs_length, 20096)


#create an array to store the pulse and noise images data
data_burst = np.zeros((1,128,128))
data_noise = np.zeros((1,128,128))
SNR_array = []
counter = 0
number_images = 10 #number of training images
number_test_images = 5
number_of_valid_pulses = 0

#Generate Pulse properties
alpha_array = np.random.normal(loc=-1.8, scale=0.5, size=number_images) #spectral index
DM_array = np.random.uniform(0,DM_max(obs_length,bottom_freq,bandwidth),size=number_images) 
log_tau_array = -6.59+0.129*np.log10(DM_array)+1.02*(np.log10(DM_array)**2)-3.86*np.log10(freqs[-1]/1000) 
idx_array=np.random.normal(loc=log_tau_array, scale=1/3)


duration_burst_array = np.random.uniform(1e-3,20e-3,size=number_images)

t_burst_min_array = -4.15*DM_array*((freqs[-1]**-2)-(freqs[0]**-2))*1000
t_burst_array = np.random.uniform(t_burst_min_array,1, size=number_images)



while (number_of_valid_pulses<number_of_valid_pulses):
    
    #create an array to store the noise for every frequency channel For every
    #iteration we need to clear it
    
    
    DM = DM_array[number_of_valid_pulses]
    t_burst = t_burst_array[number_of_valid_pulses]
    duration_burst = duration_burst_array[number_of_valid_pulses]
    amp_burst = np.random.uniform(5,25)
    t_burst_at_freqs = DM_delay(DM, freqs)
    
    flux_burst = flux_freq(amp_burst,bottom_freq,freqs,alpha_array[number_of_valid_pulses])

    model_data = flux_burst*np.exp(-0.5*(tstamp[:,None]-(t_burst_array[number_of_valid_pulses]+t_burst_at_freqs[None,:]))**2/duration_burst**2)
    
    dedispersed_data = flux_burst*np.exp(-0.5*(tstamp[:,None]-(t_burst_array[number_of_valid_pulses]))**2/duration_burst**2)
    
    conv_sig = pulse_broadening(freqs, DM, model_data, tstamp,idx_array[number_of_valid_pulses])
   
    
    noise_std_per_channel = np.random.lognormal(mean=0.2, sigma=1, size=[1,128])
    
    noise_std_per_time = np.random.lognormal(mean=0.5, sigma=0.6, size=[128,1])  

    noise = np.random.normal(loc=0., scale=1, size=(128,128))
    
    noise_matrix=noise*noise_std_per_channel*noise_std_per_time


    
    
    
    
    down_sampling_pulse = block_reduce(conv_sig,block_size=(157,4),func = np.mean)
    
    down_sampled_dedispersed = block_reduce(dedispersed_data,block_size=(157,4),func = np.mean)
    
    noisy_data = noise_matrix+down_sampling_pulse
    
    signal = np.sum(down_sampled_dedispersed, axis=1)
    total_noise = np.sum(noise_matrix, axis=1)
    SNR_actual = np.max(signal)/np.sqrt(np.mean(total_noise**2))
    
    
    if ((SNR_actual >= 5) and (SNR_actual <= 25)):
        
        SNR_array = [SNR_array, SNR_actual]
        
        number_of_valid_pulses +=1
        print(number_of_valid_pulses)
        norm_noisy_data = (noisy_data + np.abs(np.min(noisy_data)))/np.max(noisy_data + np.abs(np.min(noisy_data)))    
    
    
        norm_noisy_data = np.reshape(norm_noisy_data,(1,128,128))
        data_burst = np.vstack((data_burst,norm_noisy_data))

        
        noise_std_per_channel = np.random.lognormal(mean=0.2, sigma=1, size=[1,128]) 
        noise_std_per_time = np.random.lognormal(mean=0.5, sigma=0.6, size=[128,1])  
        noise = np.random.normal(loc=0., scale=1, size=(128,128))    
        noise_matrix=noise*noise_std_per_channel*noise_std_per_time
        norm_noise_matrix = (noise_matrix + np.abs(np.min(noise_matrix)))/np.max(noise_matrix + np.abs(np.min(noise_matrix)))

        norm_noise_matrix = np.reshape(norm_noise_matrix,(1,128,128))
        data_noise = np.vstack((data_noise,norm_noise_matrix))



SNR_array = np.array(SNR_array)

data_burst = data_burst[1::]    
data_noise = data_noise[1::]

label_pulse = np.full((len(data_burst),1), 0)    
label_noise = np.full((len(data_noise),1), 1)    

data = np.vstack((data_burst,data_noise))
label = np.vstack((label_pulse,label_noise))    



np.save(DATA_DIR + 'simulated_pulse_'+randID,data)
np.save(DATA_DIR+'labels_'+randID,label)


#Generate Pulse properties
alpha_array = np.random.normal(loc=-1.8, scale=0.5, size=number_images) #spectral index
DM_array = np.random.uniform(0,DM_max(obs_length,bottom_freq,bandwidth),size=number_images) 
log_tau_array = -6.59+0.129*np.log10(DM_array)+1.02*(np.log10(DM_array)**2)-3.86*np.log10(freqs[-1]/1000) 
idx_array=np.random.normal(loc=log_tau_array, scale=1/3)


duration_burst_array = np.random.uniform(1e-3,20e-3,size=number_images)

t_burst_min_array = -4.15*DM_array*((freqs[-1]**-2)-(freqs[0]**-2))*1000
t_burst_array = np.random.uniform(t_burst_min_array,1, size=number_images)



while (number_of_valid_pulses<number_of_valid_pulses):
    
    #create an array to store the noise for every frequency channel For every
    #iteration we need to clear it
    
    
    DM = DM_array[number_of_valid_pulses]
    t_burst = t_burst_array[number_of_valid_pulses]
    duration_burst = duration_burst_array[number_of_valid_pulses]
    amp_burst = np.random.uniform(25,50)
    t_burst_at_freqs = DM_delay(DM, freqs)
    
    flux_burst = flux_freq(amp_burst,bottom_freq,freqs,alpha_array[number_of_valid_pulses])

    model_data = flux_burst*np.exp(-0.5*(tstamp[:,None]-(t_burst_array[number_of_valid_pulses]+t_burst_at_freqs[None,:]))**2/duration_burst**2)
    
    dedispersed_data = flux_burst*np.exp(-0.5*(tstamp[:,None]-(t_burst_array[number_of_valid_pulses]))**2/duration_burst**2)
    
    conv_sig = pulse_broadening(freqs, DM, model_data, tstamp,idx_array[number_of_valid_pulses])
   
    
    noise_std_per_channel = np.random.lognormal(mean=0.2, sigma=1, size=[1,128])
    
    noise_std_per_time = np.random.lognormal(mean=0.5, sigma=0.6, size=[128,1])  

    noise = np.random.normal(loc=0., scale=1, size=(128,128))
    
    noise_matrix=noise*noise_std_per_channel*noise_std_per_time   
    
    down_sampling_pulse = block_reduce(conv_sig,block_size=(157,4),func = np.mean)
    
    down_sampled_dedispersed = block_reduce(dedispersed_data,block_size=(157,4),func = np.mean)
    
    noisy_data = noise_matrix+down_sampling_pulse
    
    signal = np.sum(down_sampled_dedispersed, axis=1)
    total_noise = np.sum(noise_matrix, axis=1)
    SNR_actual = np.max(signal)/np.sqrt(np.mean(total_noise**2))
    
    
    if ((SNR_actual >= 25) and (SNR_actual <= 50)):
        
        SNR_array = [SNR_array, SNR_actual]
        
        number_of_valid_pulses +=1
        print(number_of_valid_pulses)
        norm_noisy_data = (noisy_data + np.abs(np.min(noisy_data)))/np.max(noisy_data + np.abs(np.min(noisy_data)))    
    
    
        norm_noisy_data = np.reshape(norm_noisy_data,(1,128,128))
        data_burst = np.vstack((data_burst,norm_noisy_data))

        
        noise_std_per_channel = np.random.lognormal(mean=0.2, sigma=1, size=[1,128]) 
        noise_std_per_time = np.random.lognormal(mean=0.5, sigma=0.6, size=[128,1])  
        noise = np.random.normal(loc=0., scale=1, size=(128,128))    
        noise_matrix=noise*noise_std_per_channel*noise_std_per_time
        norm_noise_matrix = (noise_matrix + np.abs(np.min(noise_matrix)))/np.max(noise_matrix + np.abs(np.min(noise_matrix)))

        norm_noise_matrix = np.reshape(norm_noise_matrix,(1,128,128))
        data_noise = np.vstack((data_noise,norm_noise_matrix))



SNR_array = np.array(SNR_array)

data_burst = data_burst[1::]    
data_noise = data_noise[1::]

label_pulse = np.full((len(data_burst),1), 0)    
label_noise = np.full((len(data_noise),1), 1)    

data = np.vstack((data_burst,data_noise))
label = np.vstack((label_pulse,label_noise))    



np.save(DATA_DIR + 'simulated_pulse_'+randID,data)
np.save(DATA_DIR+'labels_'+randID,label)

#properties = np.vstack((DM_array,duration_burst_array,SNR_array,t_burst_array,alpha_array,idx_array))
#np.save(name_file,properties.T)

"""============================================================================

                        GENERATING TEST EXAMPLES

============================================================================"""


data_burst = np.zeros((1,128,128))
data_noise = np.zeros((1,128,128))
SNR_array = []
counter = 0
number_of_valid_pulses = 0

#Generate Pulse properties
alpha_array = np.random.normal(loc=-1.8, scale=0.5, size=number_images) #spectral index
DM_array = np.random.uniform(0,DM_max(obs_length,bottom_freq,bandwidth),size=number_images) 
log_tau_array = -6.59+0.129*np.log10(DM_array)+1.02*(np.log10(DM_array)**2)-3.86*np.log10(freqs[-1]/1000) 
idx_array=np.random.normal(loc=log_tau_array, scale=1/3)


duration_burst_array = np.random.uniform(1e-3,20e-3,size=number_images)

t_burst_min_array = -4.15*DM_array*((freqs[-1]**-2)-(freqs[0]**-2))*1000
t_burst_array = np.random.uniform(t_burst_min_array,1, size=number_images)



while (number_of_valid_pulses<number_of_valid_pulses):
    
    #create an array to store the noise for every frequency channel For every
    #iteration we need to clear it
    
    
    DM = DM_array[number_of_valid_pulses]
    t_burst = t_burst_array[number_of_valid_pulses]
    duration_burst = duration_burst_array[number_of_valid_pulses]
    amp_burst = np.random.uniform(5,25)
    t_burst_at_freqs = DM_delay(DM, freqs)
    
    flux_burst = flux_freq(amp_burst,bottom_freq,freqs,alpha_array[number_of_valid_pulses])

    model_data = flux_burst*np.exp(-0.5*(tstamp[:,None]-(t_burst_array[number_of_valid_pulses]+t_burst_at_freqs[None,:]))**2/duration_burst**2)
    
    dedispersed_data = flux_burst*np.exp(-0.5*(tstamp[:,None]-(t_burst_array[number_of_valid_pulses]))**2/duration_burst**2)
    
    conv_sig = pulse_broadening(freqs, DM, model_data, tstamp,idx_array[number_of_valid_pulses])
   
    
    noise_std_per_channel = np.random.lognormal(mean=0.2, sigma=1, size=[1,128])
    
    noise_std_per_time = np.random.lognormal(mean=0.5, sigma=0.6, size=[128,1])  

    noise = np.random.normal(loc=0., scale=1, size=(128,128))
    
    noise_matrix=noise*noise_std_per_channel*noise_std_per_time


    
    
    
    
    down_sampling_pulse = block_reduce(conv_sig,block_size=(157,4),func = np.mean)
    
    down_sampled_dedispersed = block_reduce(dedispersed_data,block_size=(157,4),func = np.mean)
    
    noisy_data = noise_matrix+down_sampling_pulse
    
    signal = np.sum(down_sampled_dedispersed, axis=1)
    total_noise = np.sum(noise_matrix, axis=1)
    SNR_actual = np.max(signal)/np.sqrt(np.mean(total_noise**2))
    
    
    if ((SNR_actual >= 6) and (SNR_actual <= 10)):
        
        SNR_array = [SNR_array, SNR_actual]
        
        number_of_valid_pulses +=1
        print(number_of_valid_pulses)
        norm_noisy_data = (noisy_data + np.abs(np.min(noisy_data)))/np.max(noisy_data + np.abs(np.min(noisy_data)))    
    
    
        norm_noisy_data = np.reshape(norm_noisy_data,(1,128,128))
        data_burst = np.vstack((data_burst,norm_noisy_data))

        
        noise_std_per_channel = np.random.lognormal(mean=0.2, sigma=1, size=[1,128]) 
        noise_std_per_time = np.random.lognormal(mean=0.5, sigma=0.6, size=[128,1])  
        noise = np.random.normal(loc=0., scale=1, size=(128,128))    
        noise_matrix=noise*noise_std_per_channel*noise_std_per_time
        norm_noise_matrix = (noise_matrix + np.abs(np.min(noise_matrix)))/np.max(noise_matrix + np.abs(np.min(noise_matrix)))

        norm_noise_matrix = np.reshape(norm_noise_matrix,(1,128,128))
        data_noise = np.vstack((data_noise,norm_noise_matrix))



SNR_array = np.array(SNR_array)

data_burst = data_burst[1::]    
data_noise = data_noise[1::]


label_pulse = np.full((len(data_burst),1), 0)    
label_noise = np.full((len(data_noise),1), 1)    

data = np.vstack((data_burst,data_noise))
label = np.vstack((label_pulse,label_noise))    

np.save(DATA_DIR + 'test_simulated_pulse_'+randID,data)
np.save(DATA_DIR+'test_labels_'+randID,label)
