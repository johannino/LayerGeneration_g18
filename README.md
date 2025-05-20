# README

This repository contains the models and code for group 18's Layer Generation project.

The folder **src** contains the whole codebase for the project. 

# Setup
1. To generate the dataset open $\texttt{GenerateCharacter.py}$. Run this script to generate layerwise data, specify in the for loop how many samples you want. It may take some time to create $5000$ samples around $10$ minutes.
2. Once the dataset is generated, you can train the different models.
    - $\texttt{UNet.py}$ contains the code for the UNet model. It is a heavy model to train, and it may take a while.
    - $\texttt{NextLayerPred.py}$ contains the code for the ViT model.
    - $\texttt{ddpm.py}$ contains the code for the diffusion model.
    - The VAR model was trained using a clone of the [original repo](https://github.com/FoundationVision/VAR).
3. After you have run the files, you will see some generated images in the $\texttt{figures}$ folder, and the models will be saved to the location you chose.
4. Finally, it is possible to evaluate the models using the $\texttt{Evaluation.py}$ or the $\texttt{EvaluationVAR.py}$ script.
