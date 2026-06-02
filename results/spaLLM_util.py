import torch
from tqdm import tqdm
import torch.nn.functional as F
from modelTriatt_Flow1 import Encoder_overall
from preprocess import adjacent_matrix_preprocessing
import random

import matplotlib.pyplot as plt

class Train_spaLLM:
    def __init__(self, 
        data,
        embedding,
        datatype = 'Brain',
        device= torch.device('cpu'),
        random_seed = 2024,
        # learning_rate=0.0001,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 5, 1, 1]
        ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.  
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.
    
        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device

        self.embedding = torch.from_numpy(embedding).to(self.device)

        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.data['adj_emb'])
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)

        self.adj_emb = self.adj['adj_emb'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        
        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_output1 = self.dim_output

        self.loss_history = []

        if self.datatype == 'Brain':
            self.epochs = 500
            print("self.weight_factors = [1,100,100]")
            self.weight_factors = [1,100,100]      # brain

        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 500
            print("self.weight_factors = [1, 1000, 1000]")
            self.weight_factors = [1, 1000, 1000]   # Ovarian

        #self.epochs = 200
        #self.weight_factors = [1,100,100]      # brain
        #self.weight_factors = [1, 1000, 1000]   # Ovarian


    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()

    def train(self, epochs=100):
        print("dim: in1, out1, in2, out2")   # 30 64 30 64
        print(self.dim_input1)
        print(self.dim_output1)

        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.embedding).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        #print("epochs: ", self.epochs)
        print("epochs: ", epochs)
        #for epoch in tqdm(range(self.epochs)):
        for epoch in tqdm(range(epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.adj_spatial_omics1, self.adj_feature_omics1, self.embedding, self.adj_emb)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])

            self.loss_rec_es = F.mse_loss(self.embedding, results['emb_recon_spa'])
            self.loss_rec_ef = F.mse_loss(self.embedding, results['emb_recon_fea'])


            print(self.loss_recon_omics1)
            print(self.loss_rec_es)
            print(self.loss_rec_ef)
            print("----------------------------")

            loss = self.weight_factors[0]*self.loss_recon_omics1 + self.weight_factors[1]*self.loss_rec_es + self.weight_factors[2]*self.loss_rec_ef
            self.loss_history.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")

        # self.plot_loss()   # print loss curve
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.adj_spatial_omics1, self.adj_feature_omics1, self.embedding, self.adj_emb)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  

        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),

                  'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
                  'alpha_att1': results['alpha_att1'].detach().cpu().numpy(),
                  'alpha_att2': results['alpha_att2'].detach().cpu().numpy()

                  }
        
        return output
    
    
    
        
    
    
      

    
        
    
    
