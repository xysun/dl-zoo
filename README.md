My deep learning code snippets. 

Examples: 
- [My solution to XOR parity challenge](xor_parity.py)
    - This is the "warmup" from [OpenAI request for research v2](https://blog.openai.com/requests-for-research-2/)
    - ![img](images/parity-accuracy.png)
- [RNN that predicts Shakespeare verse](rnn.py) (replicating [this tensorflow tutorial](https://www.tensorflow.org/tutorials/sequences/text_generation) with `Estimator` APIs, also added LSTM as a model choice)
    - Generated Shakepeare after 1000 steps of training on GRU:
     
        ```
        ROMEO:
        No, I shall be the father, that I say,
        The sense of the procled will be the seas.

        LADY ANNE:
        What says the seast that we say 'tis a thing and like to see him for the people,
        Which was the seas of t
        ```
- Demonstrate accelerated learning from batch normalization on mnist: see `batch_norm.py`
    - ![img](images/bn.png)