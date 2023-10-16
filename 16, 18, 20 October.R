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