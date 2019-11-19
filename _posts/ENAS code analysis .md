The ENAS()  is a article which is aimed at improving the effectiveness of the NAS. With the help of sharing the weight, the searching elapse time was reduced by 1000x.

Actually,the basic searching architecture of ENAS is the same as the NAS. But the search space in ENAS ,which was designed deliberately, was restricted by a DAG(从句？？？). This design idea lay the foundation to the weight para sharing. As is introduced in the article:

DAG

In the example above, we note that for each pair of nodes *j < ℓ*, there is an independent parameter matrix **W**( *ℓ,j* ). As shown in the example, by choosing the previous indices,the controller also decides which parameter matrices are used. Therefore, in ENAS, all recurrent cells in a search space share the same set of parameters. 

So,I'd like to introduce some basic methods of this article,especially for the  code implementations of related  techniques. As is proposed by ENAS, a whole convolutional network,a recurrent cell or a convolutional cell can be sampled. We'd like to lay our attention to how a convolutional network is produced with Pytorch framework.

1. Construct a huge child network which consist of 5 layers, each layer was constituted by 6 cells，each cells with 6 models in it. As is showed bellow：
   2. A LSTM controller was leveraged to sample a sequence which is passed to the child network as a strategy.
   3. Child network: DAG was got,so the architecture come out. Train and evaluate it as a common NN
   4. The sampling-training-calculating reward was tried by M times so that the ability of the controller to generate a network architecture was measured. This approach is a bit like Monte-Carlo. It is called the REINFORCE algorithm in RL domain. Here comes the key point of the article. As the child network was trained with a new DAG, it is not optimized with gradient decent algorithm from scratch，because the base frontier relation of the DAG is determined ，so the para matrix dimension is constant. Then the para which was train by the previous iteration is used also in this iteration.
   5.  As a reward is achieved,it is used to train the controller network with gradient policy as a loss value.















