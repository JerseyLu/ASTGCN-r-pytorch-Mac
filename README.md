# ASTGCN-r-pytorch-Mac
This repository contains my personal notes and summaries on the field of Traffic Prediction. Individuals who are interested at it are welcome to discuss and study together. ps: Only used for personal study!

## ASTGCN

[ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881), proposed by *Shengnan Guo*, *Youfang Lin*, *Ning Feng*, *Chao Song* and *Huaiyu Wan*, is an attention based spatial-temporal graph convolutional network model to solve traffic flow forecasting problem.

### **The Architecture**

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/232287097-d79bee2b-946f-427e-bee1-7cb8f89cde29.png" width="50%" height="50%" />
</div>

* **Spatial-Temporal Attention:**

  Propose: To capture the dynamic spatial and temporal correlations on the traffic network.

  * **Spatial Attention:**

    Spatial Attention Matrix: 
    $$
    S=V_s \cdot \sigma(( \chi_h^{(r-1)}W_1)W_2(W_3 \chi_h^{(r-1)})^T+b_s)
    $$
    
    Correlation Strength: 
    $$ \grave{S}_{i,j}= \frac{exp(S_{i,j})}{ \sum_{j=1}^N exp(S_{i,j})} $$
    
    Where $\chi_h^{r-1}=(X_1,X_2,...,X_{T_{r-1}})\in\mathbb{R}^{N\times C_{r-1}\times T_{r-1}}$ is the input of the $r^{th}$ spatial-temporal block. $C_{r-1}$ is the number of channels of the input data in the $r^{th}$ layer.

  * **Temporal Attention:**

    Tempora Attention Matrix:
    $$
    E=V_e\cdot\sigma(((\chi_h^{(r-1)})^TU_1)U_2(U_3\chi_h^{(r-1)})+b_e)
    $$
    
    Correlation Strength:
    $$ \grave{E}_{i,j} = \frac{exp(E_{i,j})}{\sum_{j=1} ^{T_{r-1}} exp(E_{i,j})} $$

  * **Spatial-Temporal Convolution:**

  Propose: To model spatial-temporal dependencies of traffic data.

  <div align="center">
  <img src="https://user-images.githubusercontent.com/104020492/232289155-2de13461-da88-4e07-b39c-cfc961e6b039.png" width="56%" height="56%" />
  </div>

  * **Graph convolution in spatial dimension:**

    In spectral graph analysis, a graph is represented by its corresponding Laplacian matrix. 

    Laplacian matrix of a graph is defined as $L = D - A$, and its normalized form is $L=I_N-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\in \mathbb{R}^{N\times N}$, where $A$ is the adjacent matrix, $I_N$ is a unit matrix, and the degree matrix $D\in \mathbb{R}^{N\times N}$ is a diagonal matrix, consisting of node degrees, $D_{ii} = \sum_j A_{jj}$. The eigenvalue decomposition of the Laplacian matrix is $L=U\Lambda U^T$ , where $\Lambda = diag([\lambda_0, ..., \lambda_{N-1}])\in \mathbb{R}^{N\times N}$ is a diagonal matrix, and $U$ is Fourier basis.
    
    **Graph Convolution Operation:**
    
    <div align="center">
    <img src="https://user-images.githubusercontent.com/104020492/232286967-25ba1325-feed-4b06-9139-3eb38e8e21fb.jpeg" width="56%" height="56%" />
    </div>
    
  * **Convolution in temporal dimension:**
  
    A standard convolution layer in the temporal dimension is further stacked to update the signal of a node by merging the information at the neighboring time slice.
    $$
    \chi_h ^{(r)}=ReLU( \phi * (ReLU(g_{ \theta * G} \hat{ \chi}_h ^{(r-1)}))) \in \mathbb{R} ^{C_r \times N \times T_r}
    $$
    
    where $*$ denotes a standard convolution operation, $\phi$  is the parameters of the temporal dimension convolution kernel, and the activation function is $ReLU$.

* **Multi-Component Fusion:**

  The final prediction:
  $$
  \hat{Y}=W_h\odot\hat{Y_h}+W_d\odot\hat{Y_d}+W_w\odot\hat{Y_w}
  $$
  
  The code is refer to [this](https://github.com/guoshnBJTU/ASTGCN-r-pytorch).  In addition, I made a small adjustment to the "DEVICE" function because the experiments I replicated were conducted on a Mac.

PS: the explanation of "search_data" function in PrepareData.py

<div align="center">
<img src="https://user-images.githubusercontent.com/104020492/232291790-40c14ee7-6a6c-47d8-97f0-32cd6fa30773.jpeg" width="56%" height="56%" />
</div>

## Configuration

Step 1: The loss function and metrics can be set in the configuration file in ./configurations

Step 2: The last three lines of the configuration file are as follows:

  ```c++
  loss_function = masked_mae
  metric_method = mask
  missing_value = 0.0
  ```

loss_function can choose 'masked_mae',  'masked_mse',  'mae',  'mse'. The loss function with a mask does not consider  missing values.

metric_method can choose 'mask', 'unmask'. The metric with a mask does not evaluate missing values.

The missing_value is the missing identification, whose default value is 0.0

## Datasets


- on PEMS04 dataset

  ```shell
  python prepareData.py --config configurations/PEMS04_astgcn.conf
  ```

- on PEMS08 dataset

  ```shell
  python prepareData.py --config configurations/PEMS08_astgcn.conf
  ```



## Train and Test

- on PEMS04 dataset

  ```shell
  python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf
  ```

- on PEMS08 dataset

  ```shell
  python train_ASTGCN_r.py --config configurations/PEMS08_astgcn.conf
  ```
