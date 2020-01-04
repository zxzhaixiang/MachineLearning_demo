# MachineLearning_demo
repos for some machine learning demo notebooks. All the notebooks should be able to run directly on Google Colab. You might need to have a Google Drive account in order to run some of the notebooks.

## Currently included notebooks

### [MetropolisHasting_sample.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/MetropolisHasting_sample.ipynb)
A quick script to sample data from a given probability distribution function using Metropolis Hasting Method. The MH method is a Markov Chain Monte Carlo based algorithm, which can generate unbiased sample from a given PDF without the need of computing CDF. The PDF also does not need to be uniformed. This makes MH method very suitable to sample from high dimensional data. The notebook includes a quick example of sampling points from a standard Gaussian method. THe pdf function can be replaced by customized functions.

### [node2vec.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/node2vec.ipynb)
A simple neural network implementation to embede nodes in a undirectional graph. node2vec is the simplest implementation of a graph neural network. node2vec aims to embed nodes in a undirectional graph such that connected nodes have similar embedding (their cosine similarity or dot product is closed to 1). Implemented with Pytorch.

### [CollaborativeFiltering.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/CollaborativeFiltering.ipynb)
A comprehensive study of neural collaborative filtering algorithm on MovieLens-latest 27M dataset. To reduce training time, a dense subset of user x item interaction are selected. The model can be trained on Colab within 5 min to achieve very decent performance. The latent space of movie embedding are particularly studied. Implemented with Pytorch

### [boosting_tree_quantile_regression.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/boosting_tree_quantile_regression.ipynb)
The notebook shows how to perform quantile regression using a few popular boosting tree packages, including XGBoost, lightGBM, and Scikit-Learn's GradientBoostingRegressor. Quantile regression is helpful to estimate the uncertainty of a machine learning prediction. A customized "soft huber quantile loss" (skewed L1 loss) is used to replace the default loss function in XGBoost and lightGBM. The customized quantile loss shows comparable performance to the native quantile regression built inside lightGBM and Scikit-learn. XGboost currently doesn't suppor quantile regression natively, so feeding a customized loss function is the only way to achieve the goal so far.

### [gaussian_processing.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/gaussian_processing.ipynb)
A numpy implementation of Gaussian processing algorithm. Gaussian processing is a machine learning (or more a conditional function distribution function estimation) algorithm that is very intuitive and powerful to learn smooth functions. Gaussian processing is the foundation for Bayesian Optimization. In the notebook, a simple radius based kernel function (RBF) is used.

### [BayesianOptimization.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/BayesianOptimization.ipynb)
A comprehensive workflow of using Baysesian Optimization algorithm to search for global maximum of functions. Baysesian Optimization is a power non-gradient based optimization algorithm, which utilizes Gaussian Processing algorithm to effeciently estimate and sample observations from the functions. Bayesian Optimization is most useful when evaluating the target function is expensive or time consuming, such as hyperparameter tuning, geological modeling for mining/fossil fuel extraction, etc.

In the notebook, several acquisition functions are provided, including simple mean/variance combination (exploration/exploitation tradeoff), improved probability, and expected improvement. Both 1D and 2D cases are tested. Higher dimension cases should also work with the same optimization function, just be cautious on scaling of different dimension.

### [DQN_Reinforcement_Learning_on_GYM_Atari.ipynb](https://github.com/zxzhaixiang/MachineLearning_demo/DQN_Reinforcement_Learning_on_GYM_Atari.ipynb)
Probably the most fun notebook! Use deep Q network learning to learn how to place Atari game by only observing the game's output screen! The example used here is Breakout, and OpenAI's GYM Atari environment is used. The Q function is formulated as a small convolutional neural network which takes the screen output of the Atari game as input (usually a stack of a few consecutive frames) and then outputs the expected total reward for each action. In case of breakout, the action includes NOOP, FIRE (to start a game), LEFT and RIGHT. On Colab, with a few hour training, after seen something like 10,000 episodes of the game, the algorithm is able to learn the strategy automatically and achieves a high score above 50! A few examples are also included in the repos, such as this one
![DQN playing Breakout](https://github.com/zxzhaixiang/MachineLearning_demo/blob/master/BreakoutNoFrameskip-v4-1200-496frame_76.0.gif)

## A trick to prevent Google Colab Timeout
If you leave the Colab notebook to run for too long without attention, Colab might decide to timeout and you will have to reconnect the the runtime. You might lose your progress. To avoid this and be able to use Colab for a straight 12hr (granted by Google), you can do the following trick in the browser:

- Press F12 in the browser to open JavaScript console
- Paste the following code to console
```javascript
function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
```
This will keep Colab active until your session is ended.
