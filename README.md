# PyTorch implementation of Spatial Transformer Networks applied to the CelebA dataset
*Author: Esteban Leiva Montenegro*

Spatial Transformer Networks (STNs) are a powerful class of neural networks that enable spatial manipulation of input data within the network architecture itself[^1]. With PyTorch, implementing STNs becomes more accessible, offering flexibility and efficiency. In this context, leveraging the CelebA dataset[^2], renowned for its diverse facial images, provides a rich environment for exploring STNs capabilities in tasks like gender recognition implemented with a ResNet101[^3].

<!-- >[!NOTE]
>If you wish to skip the usage guide and directly access the results, you can do so by following this link: [Experimental design](#2-experimental-design) -->

## 1. Usage 
### 1.1 Download the dataset
Note that the ~200,000 CelebA face image dataset is relatively large (~1.3 Gb). The download link provided below was provided by the author on the official CelebA website at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.

1. Download and unzip the file `img_align_celeba.zip`, which contains the images in jpeg format.

2. Download the `list_attr_celeba.txt` file, which contains the class labels

3. Download the `list_eval_partition.txt` file, which contains training/validation/test partitioning info
### 
4. Move the downloaded files to *./celeba* directory
    ```
        📁 Root
        ├── 📁 celeba
        │   ├── 📜 img_align_celeba.zip
        │   ├── 📜 list_attr_celeba.txt
        │   └── 📜 list_eval_partition.txt
        ├── 📁 src
        │   └── ...
        └── 📜 README.md
    ```
### 1.2 Install dependencies
You can install all the project dependencies running the following command
`pip install -r requirements.txt`
### 1.3. Run ./src/app.py
<!-- >[!NOTE]
>If Docker engine isn't installed on your system, please refer to the official installation guide [here](https://docs.docker.com/). -->

>[!TIP]
>To adjust hyperparameters, you can modify the **./src/config.json** file. Note: *Ensure that the BATCH_SIZE in the loaded data and the model BATCH_SIZE remain consistent.*

1. Run the following command `python ./src/app.py`

<!-- ## 2 Experimental design 
### 2.1 Hyperparameters

### 2.2 Training

### 2.3 Results -->
 
## References 
[^1]: Jaderberg, M., Simonyan, K., Zisserman, A., & Kavukcuoglu, K. (2016). "Spatial Transformer Networks." *arXiv preprint arXiv:1506.02025 [cs.CV]*, version 3, February 4, 2016. [Online]. Available: https://arxiv.org/abs/1506.02025
[^2]: Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). "Deep Learning Face Attributes in the Wild." In *Proceedings of International Conference on Computer Vision (ICCV)*, December 2015.
[^3]: Raschka, S. (2019). "CNN-ResNet101-CelebA." [Online]. Available: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb


