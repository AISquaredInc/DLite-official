# DLite Official Repository

This repository contains code to train `DLite` and `DLiteV2`, large language models from AI Squared. The repository can also be modified to train other models on the same datasets.

The `DLite` family of models primarily demonstrates that even though model size does have an impact on performance, smaller models can perform relatively well on a wide range of tasks, including acting as a chatbot.

The `DLite` family of models are not a state-of-the-art language models, and they are not expected to compete with more modern language models trained on more comprehensive datasets. Instead, the main contribution of `DLite` is that such a small model which is (at the time of this work) nearly four years old can be effectively trained to act as a chat-based agent.

**DLiteV1 is intended only for research purposes and is not licensed for commercial use.**

**DLiteV2 is licensed for commercial use.**

## Training DLiteV2

There are two ways to train models using this repository. The first is to simply run the `train_dlite.ipynb` notebook in the top level of this repository. Additionally, if you would like to run the training script from the command line, the `train.py` file in the `train` directory of this repository is complete with a command line interface.

By altering the base model and the training dataset (either `tatsu-lab/alpaca` or `aisquared/databricks-dolly-15k` are acceptable), users are able to train any version of their own `DLite` model.

## Limitations

*DLite is an experimental technology and is not designed for use in any environment without significant testing and safety consideration. Furthermore, the model can sometimes exhibit undesired behaviors. Some of these behaviors include, but are not limited to: factual inaccuracies, biases, offensive responses, toxicity, and hallucinations. Just as with any other LLM, we advise users of this technology to exercise good judgment when applying this technology.*
