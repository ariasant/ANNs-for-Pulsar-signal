# ANNs-for-Pulsar-signal
Two artificial neural networks for the identification of single-pulse signal in radio observation, and the regression of the associated dispersion measure.
Work done as part of my Master's thesis in collaboration with Sergio Belmonte Diaz. More information are available in the reports in the repository.
Structure of the repository:

- Simulate Pulse Images.py 
  python script to simulate realistic time-frequency radio observations of single-pulse signals from pulsars, and of noise.
- Pulse Classification Algorithm.py
  Implementation and training of a convolutional neural network for the classification of single-pulse signal and noise time-frequency images.
- DM Regression Algorithm.py
  ResNet-like artificial neural network for estimating the dispersion measure associated to the time-frequency images containing single-pulse signal. 
 
