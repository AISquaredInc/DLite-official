# DLite V2

Code used to train `DLiteV2`, a large language model from AI Squared

`DLiteV2` primarily demonstrates that even though model size does have an impact on performance, smaller models can perform relatively well on a wide range of tasks, including acting as a chatbot. We have utilized the `databricks-dolly-15k` dataset to fine-tune the 124 million parameter GPT-2 model from OpenAI to create `DLiteV2`.

`DLiteV2` is not a state-of-the-art language model, and it is not expected to compete with more modern language models trained on more comprehensive datasets. Instead, the main contribution of `DLiteV2` is that such a small model which is (at the time of this work) nearly four years old can be effectively trained to act as a chat-based agent.

**DLiteV2 is intended only for research purposes and is not licensed for commercial use.**

## Training DLiteV2

There are two ways to train `DLiteV2` using this repository. The first is to simply run the `train_dlite.ipynb` notebook in the top level of this repository. Additionally, if you would like to run the training script from the command line, the `train.py` file in the `train` directory of this repository is complete with a command line interface.

## Limitations

*DLiteV2 is an experimental technology and is not designed for use in any environment other than for research purposes. Furthermore, the model can sometimes exhibit undesired behaviors. Some of these behaviors include, but are not limited to: factual inaccuracies, biases, offensive responses, toxicity, and hallucinations. Just as with any other LLM, we advise users of this technology to exercise good judgment when applying this technology.*
