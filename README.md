## Get started with JAX! :computer: :zap:

The goal of this repo is to make it easier to get started with [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Haiku](https://github.com/deepmind/dm-haiku)!

`JAX` ecosystem is becoming an increasingly popular alternative to `PyTorch` and `TensorFlow`. :sunglasses:

<br/>
<br/>

<p align="center">
<img src="readme_pics/jax_logo.png" width="300"/>
</p>

<br/>
<br/>

*Note: I'm only going to recommend content that I've personally analyzed and found useful here. 
If you want a comprehensive list check out the [awesome-jax repo](https://github.com/n2cholas/awesome-jax).*

## Table of Contents
  * [Machine Learning with JAX](#my-machine-learning-with-jax-tutorials)
    + [Tutorial #1: From Zero to Hero](#tutorial-1-from-zero-to-hero)
    + [Tutorial #2: From Hero to Hero Pro+](#tutorial-2-from-hero-to-heropro)
    + [Tutorial #3: Coding a Neural Network from Scratch in Pure JAX](#tutorial-3-building-a-neural-network-from-scratch)
    + [Tutorial #4: Flax From Zero to Hero](#tutorial-4-machine-learning-with-flax---from-zero-to-hero)
    + [Tutorial #5: Haiku From Zero to Hero (coming soon)](#tutorial-5-coming-up-machine-learning-with-haiku---from-zero-to-hero)
  * [Other useful JAX resources](#other-useful-content)

## My Machine Learning with JAX Tutorials

*Tip on how to use notebooks: just open the notebook directly in Google Colab 
(you'll see a button on top of the Jupyter file which will direct you to Colab). 
This way you can avoid having to setup the Python env! (This was especially convenient for me since I'm on Windows which is still not supported)*

### Tutorial #1: From Zero to Hero

In this video, we start from the basics and then gradually dig into the nitty-gritty details
of `jit`, `grad`, `vmap`, and various other idiosyncrasies of JAX.

[YouTube Video (Tutorial #1)](https://youtu.be/SstuvS-tVc0) <br/>
[Accompanying Jupyter Notebook](https://github.com/gordicaleksa/get-started-with-JAX/blob/main/Tutorial_1_JAX_Zero2Hero_Colab.ipynb) <br/>

<p align="left">
<a href="https://www.youtube.com/watch?v=SstuvS-tVc0" target="_blank"><img src="https://img.youtube.com/vi/SstuvS-tVc0/0.jpg" 
alt="JAX from zero to hero!" width="480" height="360" border="10" /></a>
</p>

### Tutorial #2: From Hero to HeroPro+

In this video, we learn all additional components needed to train ML models (such as NNs) on multiple machines!
We'll train a simple MLP model and we'll even train an ML model on 8 TPU cores!

[YouTube Video (Tutorial #2)](https://www.youtube.com/watch?v=CQQaifxuFcs) <br/>
[Accompanying Jupyter Notebook](https://github.com/gordicaleksa/get-started-with-JAX/blob/main/Tutorial_2_JAX_HeroPro%2B_Colab.ipynb) <br/>

<p align="left">
<a href="https://www.youtube.com/watch?v=CQQaifxuFcs" target="_blank"><img src="https://img.youtube.com/vi/CQQaifxuFcs/0.jpg" 
alt="JAX from Hero to HeroPro+!" width="480" height="360" border="10" /></a>
</p>

### Tutorial #3: Building a Neural Network from Scratch

Watch me code a Neural Network from scratch! :partying_face: In this 3rd video of the JAX tutorials series.

In this video, I build an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) and train it as a classifier on MNIST 
using PyTorch's data loader (although it's trivial to use a more complex dataset) - all this in "pure" JAX (no Flax/Haiku/Optax).

I then do an additional analysis:
* Visualize MLP's learned weights
* Visualize embeddings of a batch of images using t-SNE
* Finally, I analyze whether we have too many dead ReLU neurons in our network

[YouTube Video (Tutorial #3)](https://www.youtube.com/watch?v=6_PqUPxRmjY) <br/>
[Accompanying Jupyter Notebook](https://github.com/gordicaleksa/get-started-with-JAX/blob/main/Tutorial_3_JAX_Neural_Network_from_Scratch_Colab.ipynb) (Note: I'll soon refactor it but I'll link the original)<br/>

<p align="left">
<a href="https://www.youtube.com/watch?v=6_PqUPxRmjY" target="_blank"><img src="https://img.youtube.com/vi/6_PqUPxRmjY/0.jpg" 
alt="Building a Neural Network from Scratch in pure JAX!" width="480" height="360" border="10" /></a>
</p>

---

### Tutorial #4: Machine Learning with Flax - From Zero to Hero

In this video, I cover everything you need to know to get started with [Flax](https://github.com/google/flax)!

We cover `init`, `apply`, `TrainState`, etc. and other idiosyncrasies like the usage of `mutable` and `rngs` keywords.

[YouTube Video (Tutorial #4)](https://www.youtube.com/watch?v=5eUSmJvK8WA) <br/>
[Accompanying Jupyter Notebook](https://github.com/gordicaleksa/get-started-with-JAX/blob/main/Tutorial_4_Flax_Zero2Hero_Colab.ipynb) <br/>

<p align="left">
<a href="https://www.youtube.com/watch?v=5eUSmJvK8WA" target="_blank"><img src="https://img.youtube.com/vi/5eUSmJvK8WA/0.jpg" 
alt="Flax from Zero to Hero!" width="480" height="360" border="10" /></a>
</p>

---

### Tutorial #5 (coming up): Machine Learning with Haiku - From Zero to Hero

todo

## Other useful content

Aside from the [official docs](https://jax.readthedocs.io/) here are some resources that helped me.

### Videos

* [Introduction to JAX](https://www.youtube.com/watch?v=0mVmRHMaOJ4&ab_channel=GoogleCloudTech) (gives a very high-level overview)
* [JAX: Accelerated Machine Learning Research | SciPy 2020 | VanderPlas](https://www.youtube.com/watch?v=z-WSrQDXkuM&ab_channel=Enthought) (many more details)
* [NeurIPS 2020: JAX Ecosystem Meetup](https://www.youtube.com/watch?v=iDxJxIyzSiM&t=1s&ab_channel=DeepMind) (DeepMind team about the ecosystem of libs around JAX)
* [Introduction to JAX for Machine Learning and More](https://www.youtube.com/watch?v=QkmKfzxbCLQ&ab_channel=UWaterlooDataScience) (nice, hands-on workshop)
* [Day 1 Talks: JAX, Flax & Transformers | HuggingFace](https://www.youtube.com/watch?v=fuAyUQcVzTY&ab_channel=HuggingFace) (all 4 talks are good)
* [Day 2 Talks: JAX, Flax & Transformers | HuggingFace](https://www.youtube.com/watch?v=__eG63ZP_5g&ab_channel=HuggingFace) (only the first 2 talks are relevant)

### Blogs

* [Using JAX to accelerate our research | DeepMind](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) (similar info as the NeuroIPS 2020 video)
* [You don't know JAX | Colin Raffel](https://colinraffel.com/blog/you-don-t-know-jax.html)

## Acknowledgements

* The notebooks were heavily inspired by the official [JAX](https://jax.readthedocs.io/), [Flax](https://flax.readthedocs.io/en/latest/), and [Haiku](https://dm-haiku.readthedocs.io/en/latest/) docs.

## Citation

If you find this content useful, please cite the following:

```
@misc{Gordic2021GetStartedWithJAX,
  author = {Gordić, Aleksa},
  title = {Get started with JAX},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/get-started-with-JAX}},
}
```

## Connect With Me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:
* Subscribing to my YouTube channel [The AI Epiphany](https://www.youtube.com/c/TheAiEpiphany) :bell:
* Follow me on [LinkedIn](https://www.linkedin.com/in/aleksagordic/) and [Twitter](https://twitter.com/gordic_aleksa) :bulb:
* Follow me on [Medium](https://gordicaleksa.medium.com/) :books: :heart:
* Join the [Discord](https://discord.gg/peBrCpheKE) community! :family:

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/get-started-with-JAX/blob/master/LICENCE)