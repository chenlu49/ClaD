# Project Description

This project provides the code for ClaD model training and evaluation on misogyny data, split into two main files:

## File Structure
- **`ClaD_misogyny.py`**  
  - Contains the model definition, training code, and logic for saving model parameters.  
  - Key features:
    - Load pre-trained models (e.g., XLNet).  
    - Build task-specific structures (supports adding extra layers, freezing certain layers, etc.).  
    - Define loss functions (e.g., Mahalanobis mean loss) and perform training.  
    - Save the trained model weights.  

- **`evaluation_xlnet_misogyny.ipynb`**  
  - Used for model evaluation and analysis.  
  - Key features:
    - Load the trained model weights.  
    - Run predictions on test data.  
    - Compute various evaluation metrics (e.g., accuracy, F1 score, Mahalanobis distance).  
    - Support result visualization for better performance analysis.  


