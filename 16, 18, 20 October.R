# 16 October - Classification Random Forests 

# to build a tree - looks through every x and every possible split point along that variable 
#   - considers what the loss function is for each split; keeps what's minimized 
# how do you build multiple trees? 
#   - tree to forest 
#   - how does this ensemble come about: bootstrapping !! 
# how do we believe an ecologically diverse forest? 
#   - the bootstrapping naturall gives us different samples 
#      - randomly selecting explanatory variables to be in each tree to be potential splits 
# because this is classification, not a continuous variable, the majority rules here 
#    - there's no 'averaging' like w continuous 
# Gini Index (Gini impurity) is used to grow tree 
#   - ends up being the variance of a bernoulli distribution 
#   - we want p to either be very close to 0 or very close to 1 

# 18 October - Naive Bayes  

# Bayes Theorem 
#   - a way of switching the conditioning in probability 
#   - we want the probability of it being a 1, or a yes, etc. given explanatory variables 
#       - bayes is telling us to flip it around
#       - we need to know: 
#           - prob(x) given y = 1 ; likelihood - probability density/mass functions 
#           - marginal prob of y = 1 
#           - marginal of all the x's 
# lets look at two densities: 
#   - one to the yes, one to the know 
#   - for a new pred point, does it fit under the yes or the no better? = bayes theorem here 

# "Naive" assumption - all the features are independent (all of the explanatory variables)
#   = factored into individual probabilities 
#   - basically means that for one x, we're doing the plot with the two density plots 

# What do we know about the x's given y = 1? 
#   - what proportion are green that has a response = 1? - for categories! 
#   - in my zero group, what proportion are green blue black etc. 
#   
# Kernel Density Estimation - creates a density plot for us and is a function of smoothness 
#   - controls the picture of the densities 

# does this x im trying to predict belong with the zeroes or the ones = basic of all this 
#  - then weight by how common 1s and 0s are; here we're erring on the side of the 1

# laplace constant - bayes involves prior distributions 
#  - basically how much prior info we want to include in prediction 
#  - laplace acts as a prior 
#      - really big = evens everything out 

# prameters = laplace and smoothness 

# Advantages: 
#  - fast to train and fast to work 
#  - not many parameters 
#  - results are typically interpretable; given in probability 
# Disadvantages: 
#  - 'naive' assumption is rarely true; typically need to alter our data for indepenedence 
#  - models don't perform as well in complex situation; won't beat out random forest 
#  - unimportant features can mess it up 

# calculating the prob of y = 1 given all the x's with a flipped calculation 
# gives us to densities to predict 



# 20 October - K Nearest Neighbors 

# choose a point - star in diagram; classify into class A or class B based on the nearest neighbors 
#    - can change depending on how many 'neighbors' there are = majority voting 
#    - called a 'lazy learner': doesn't perform calculations/training until it's time to predict 

# best to choose K using CV 
#   - start w srt(n), but tuning is best 
#   - if K is too large, it will underfit; if it's too small it will overfit 

# no assumptions need to be made about the data 
# computationally expensive 
# Curse of Dimensionality - more features than data 
#   - solve w variable selection methods 
#   - PCA: project onto a smaller subspace 

# most common way to define distance b/t two points: euclidian distance 





