import model_cnn as M
import model_nlp_cnn as MNLP
import model_nlp as M_Simp
import model_attention as M_Attn
import torch

class Model_Context_Manager():
    def __init__(self, option):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_selector = lambda option: (
                                            M.NNModel().to(self.device) if option == 0 else
                                            MNLP.NNModelNLP().to(self.device) if option == 1 else
                                            M_Simp.NN_Simple().to(self.device) if option == 2 else
                                            M_Attn.NNAttention((3, 224, 224)).to(self.device) if option == 3 else None
                                        )
        self._model = model_selector(option)
    @property
    def model(self):
        return self._model 