# LayerGeneration_g18

This repository contains the models and solutions for group 18's answers to the Layer Generation project.

The folder **src** contains the whole codebase for the project. 

# Setup
- To generate the dataset open $\texttt{GenerateCharacter.py}$. Run this script to generate layerwise data, specify in the for loop how many samples you want. It may take some time to create $5000$ samples around $10$ minutes.
- Once this is done, run the python files which results you want to reproduce.
    - $\texttt{UNet.py}$ contains the code for the UNet model. It is a heavy model to train, and it may take a while.
    - $\texttt{NextLayerPred.py}$ contains the code for the ViT model.
    - $\texttt{ddpm.py}$ contains the code for the diffusion model.

After you have run the files, you will see your results show up in $\texttt{figures}$ folder.

After having run the files and saved some good models, it is possible to see evaluate the models using the $\texttt{Evaluation.py}$ script.

