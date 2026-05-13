"""Loss Functiions for Classification Tasks"""

import numpy as np

class LossFunctions:
    """Loss Functions for Classification Tasks"""
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        """Cross-Entropy Loss Function"""
        m = y_true.shape[0]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    

