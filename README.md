# Image Generation with Generative Adversarial Network

# KL Divergence
## https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
## https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/02b4614b-c5f3-4150-ae61-953585d263ef)
##
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/83b5e5d0-549d-4829-a213-894af2da3a1c)
##
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/223454e6-0494-45b8-8083-743cf77e74b2)

# How to sample random numbers from a distribution
## https://www.quora.com/How-do-you-sample-from-a-probability-density-function-PDF
## https://medium.com/mti-technology/how-to-generate-gaussian-samples-347c391b7959

## f is pdf, F is cdf
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/aa719b88-266e-4c9e-8b9b-fe45af3663fe)
##
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/438109b3-1122-4b66-b3db-2569d11a794c)
##
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/37d53f1d-b8d7-4da7-a745-13a900fd0c48)



# Generate a sample - height 

Generate a height from 200 height samples following Pdata(x):

In height space, there are infinite height distributions. The below distribution is simply one of them
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/b28b5457-6725-4975-9939-19c473060a57)






# Modeling - GMM

Pg(x, θ)


K = 1,2 ：1 - boy，2 - girl
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/24532010-2a0e-4501-b103-e797671e8d3d)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/91c6838f-1e45-41af-a2a3-b49e5ac40a4e)


three parameters for every model: mean, deviation, and weights
U1，θ1，a1 (Probability of Gaussian Model 1 - Boys)；
U2，θ2，a2 (Probability of Gaussian Model 2 - Girls)；

a1 = 0.5，a2 = 0.5  ( 100 boys, 100 girls )
a1 = 0.6，a2 = 0.4  ( 120 boys, 80 girls )




# MLE ![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/dbcfd248-01c1-44ff-a3e9-dd74f2f53e07)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/db24ae97-c8c7-4c24-8563-4c3ca0185253)



#  Generate a sample - image  (400x400x1 channel)

Generate an image from 20000 image samples.

The image space contains unlimited images ( following unlimited distributions );

The 20000 samples follow a specific distribution - Pdata(x);


# Modeling - GMM
 
Pg(x; θ): each pixel follows a GMM (n x 3 parameters ), totally 160000 GMMs; 

If every GMM have two distributions, there are 6 variables, there 0.96 million parameters




# Old Solution
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/c46203ea-31ac-4396-a31f-093eb30673fa)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/e8673b99-9f9b-4818-87ab-bd6bd710136b)
It is impossible to define the model directly.



# MLE - minimize the KLD
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/bc805fb9-254c-46b9-ac45-59e9fd9e90b4)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/3ef728db-1856-46aa-903c-781ac5ac5271)



# New Solution

1)Introduce Z, a latent 32-D (or more) variable;
Each dimension follows the unit guassian - N(0,1) and represents a feature - noise, eye, height, color and hair (assuming we geneate human images).

2)Train a generator, G(z) = x.

3)Pg(x; θ) can approximate Pdata(x).
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/3884b2ee-1017-430e-989b-3d0094cbc1ff)



# Explicit Probabilistic Model vs. Implicit Probabilistic Model
https://datascience.stackexchange.com/questions/27953/difference-between-explicit-and-implicit-density-with-and-without-the-relation-t
https://www.cs.cmu.edu/~epxing/Class/10708-17/notes-17/10708-scribe-lecture19.pdf
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/5abc9b36-5065-4fc8-b815-6714269fd816)



# Nash Equilibrium
https://www.investopedia.com/terms/n/nash-equilibrium.asp![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/f2f2fbda-1d63-4c66-b7d6-a88c3204d704)


# GAN - Generative Adversial Networks
https://towardsdatascience.com/intuitive-introduction-to-generative-adversarial-networks-gans-230e76f973a9



# GAN  - DCGAN, Deep Convolutioanl GAN

Fix D, update G;
Fix G, update D;

![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/d8ea99b0-7034-4943-9ca1-5b5511a71d1b)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/cbec56dc-dc01-470e-ac85-7ebc3574de65)



# Transposed Convolution
https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/85587f85-d7fd-49a7-a5b6-738c23cd120c)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/53cca4b1-97c5-4df5-9c25-0d136f7014e9)
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/ff5a63fe-f3f9-48fa-a34a-a57fb29c96f9)


# GAN Math
https://towardsdatascience.com/decoding-the-basic-math-in-gan-simplified-version-6fb6b079793
https://jaketae.github.io/study/gan-math/
![image](https://github.com/yinanericxue/Image-Generation-with-Generative-Adversarial-Network/assets/102645083/40ad7151-ae47-4c18-b288-ffeafcd24ac9)



