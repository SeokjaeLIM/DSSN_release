# DSSN_release
This repository is a Pytorch implementation of the paper [**"Deep Spectral-Spatial Network for Single Image Deblurring"**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9094296)

Seokjae Lim, Jin Kim and [Wonjun Kim](https://sites.google.com/site/kudcvlab)  
IEEE Signal Processing Letters (Early Access)

When using this code in your research, please cite the following paper:  

Seokjae Lim, Jin Kim and Wonjun Kim, **"Deep Spectral-Spatial Network for Single Image Deblurring,"** **IEEE Signal Processing Letters (Early Access)**.

### Model architecture
![examples](./examples/network.png)

### Experimental results with state-of-the art methods on the GOPRO dataset
![examples](./examples/results.png)

### Experimental results with state-of-the art methods on the Köhler dataset
![examples](./examples/results2.PNG)
Several results of single image deblurring. First column : input blurry images selected from the Köhler dataset. Second column : deblurring results by Nah et al. Third column : deblurring result by Zhang et al. Fourth column : deblurring results by the proposed method. Note that all experiments are conducted with parameters, which are trained based on the GOPRO dataset, without any modification.
