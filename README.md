# QA with R-NET

The model is mostly based on R-NET:

["R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS" - Natural Language Computing Group, Microsoft Research Asia](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

As of 6th December 2017, a version of R-NET (with ensembles) acheives the highest position in [SQuAD leaderboards.](https://rajpurkar.github.io/SQuAD-explorer/) 

For this model I used pre-trained GloVe embeddings of 100 dimensions from:  
https://nlp.stanford.edu/projects/glove/

I skipped character embedding. I used to the positional encoding as used in ["End-To-End Memory Networks" (- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus)](https://arxiv.org/abs/1503.08895), for sentence embedding of the facts. That is, the network works with sentential representations of facts, instead of working with word level representations of the facts. 

Only 1-layered Bi-GRUs were used for encoding questions, passages and other works. The original implementation used 3-layered Bi-GRU for certain tasks. 

Instead of a pointer network (which was used to predict the span of the original passage to work with SQuAD), I used a single GRU upon the question-aware self attended document representation (the output of the self-matching attention layer) and linearly transformed the final hidden state to get the probability distribution for the single word answer. 

I used the transpose of the embedding matrix to linearly transform the final hidden state of the output layer into the probability distribution. Using embedding matrix for the final conversion to probability distribution seems to usually speed up training, sometimes quite substantially, without bringing any apparent detrimental effects. 

I trained and tested the model on the induction tasks of [bAbi 10k dataset](https://research.fb.com/downloads/babi/). The accuracy isn't too different from [my DMN+ implementation](https://github.com/JRC1995/Dynamic-Memory-Network-Plus) for this particular task when used with these specific settings. 

Some other hyperparameters are different. 

This is a rough and quick implementation, I made up in a hour or two. There may be some issues that I have overlooked.



