# CBCT-to-CT

This project is code for deep learning-based CBCT-to-CT translation, implemented in PyTorch using flow matching models and denoising diffusion probabilistic models.

The project includes three sections: 
1) DataPrepare: Convert CT and CBCT in DICOM format to the easily readable .npy format, and generate simulated CBCT based on CT data.
2) sCT: CBCT-to-CT conversion using deep learning
3) Evaluate: Assessment of model performance.

# How to run the code

The code for each section can be run independently. After modifying configuration parameters such as file paths, directly run main.py in the src folder.


