MATLAB package for the following pansharpening algorithms:
    a) PNN  [Masi et al. (2016)] 
    b) PNN+ [Scarpa et al. (2018)]

References
        [Masi2016]      G. Masi, D. Cozzolino, L. Verdoliva and G. Scarpa 
                        “Pansharpening by Convolutional Neural Networks” 
                        Remote Sensing, 
                        vol. , no. , pp. , July 2016.

        [Scarpa2018]    G. Scarpa, S. Vitale and D. Cozzolino, 
                        "Target-Adaptive CNN-Based Pansharpening" 
                        IEEE Transactions on Geoscience and Remote Sensing, 
                        vol. 56, no. 9, pp. 5443-5457, Sept. 2018.

Main functions to call:  
        "PNN" and "PNNplus"

How to install:
        Add folder path to your Matlab environment and go:
           >> addpath('<folder location on your pc>/PNNplus_Matlab_code/');

System requirements:
        Matlab2018b or higher versions, with deep learning toolboxes.
        The running on previous Matlab versions is not guaranteed.

Execution Environment:
        The code runs on GPU when available or on CPU, otherwise.
    WARNING:    In the CPU case execution slows down. It is recommended to 
                limit to a few iterations (TF_epochs<10) or skip 
                (FT_epochs=0) the fine-tuning for PNN+ in this case.

Test:
        To test the code move to the companion folder 'TestPNNplus', that 
        includes three sample images (Ikonos, GeoEye1 and WV2). 
	Uncomment one row in “testPNNplus.m” to select the sample and run it.
	The following alternative pansharpened images will be provided and saved (.mat):

	a) PNN+ with fine tuning (50 iterations): 	“img<sample>_PNNplus.mat”
	b) PNN+ without fintuning: 			“img<sample>_PNNplus_noFT.mat”
	c) PNN with additional input bands: 		“img<sample>_PNN.mat”
	c) PNN without additional input bands: 	“img<sample>_PNN_noIDX.mat”
