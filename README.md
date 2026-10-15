## About The Project

A two-part deep learning research project developed as part of Singapore Polytechnic's **Deep Learning with Applications (ST1504)** curriculum. 

The project evaluates advanced **Generative Deep Learning (GANs)** for synthetic image synthesis and **Deep Reinforcement Learning (RL)** for continuous physical control.

### Project Components

#### Part A: Generative Adversarial Networks (GANs)
* **Architectures Evaluated**: Implemented and benchmarked **DCGAN (Deep Convolutional GAN)**, **cGAN (Conditional GAN)**, and **WGAN-GP (Wasserstein GAN with Gradient Penalty)**.
* **Training Stability & Mode Collapse Mitigation**: Leveraged the Earth Mover's Distance and Lipschitz continuity constraints via gradient penalty to stabilize generator-discriminator adversarial convergence.
* **Evaluation & Metrics**: Tracked loss curves, synthetic visual fidelity, and generated output metrics across training epochs.

#### Part B: Deep Reinforcement Learning (RL)
* **Continuous Control**: Designed and trained an autonomous RL policy agent to solve the **OpenAI Gym Pendulum** benchmark environment.
* **Policy Optimization**: Implemented the actor-critic/policy iteration loop, collecting environment transition tuples $(s, a, r, s')$ and optimizing cumulative discounted return.
* **Trajectory Rendering**: Recorded agent performance and rollout dynamics using OpenCV video rendering.

### Tech Stack
* **Frameworks**: TensorFlow / Keras, PyTorch
* **Reinforcement Learning**: OpenAI Gym (`gym==0.17.3`)
* **Computer Vision & Scientific Computing**: OpenCV, NumPy, Matplotlib, Pandas
