from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from datasets import load_dataset, Dataset
from functools import partial
from tqdm import tqdm
import numpy as np
import re

DATASET = 'aisquared/databricks-dolly-15k'
MODEL_ID = 'EleutherAI/gpt-neo-125m'
END_KEY = '### End'
INSTRUCTION_KEY = '### Instruction:'
INPUT_KEY = '### Input:'
RESPONSE_KEY = '### Response:\n'
SEED = 42

PROMPT_WITH_INPUT = """The following is an instruction that describes a task, paired with an input that provides further context. Write a response that completes this task.

%s
{instruction}

%s
{input}

%s""" % (INSTRUCTION_KEY, INPUT_KEY, RESPONSE_KEY)

PROMPT = """The following is an instruction that describes a task. Write a response that appropriately completes the request.

%s
{instruction}

%s""" % (INSTRUCTION_KEY, RESPONSE_KEY)

def load_model_and_tokenizer(location):
    """
    Load the model and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(
        location,
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        location,
        trust_remote_code = True
    )
    return model, tokenizer

def create_response(
        instruction,
        model,
        tokenizer,
        do_sample = True,
        max_new_tokens = 256,
        top_p = 0.92,
        top_k = 0,
        device = None,
        **kwargs
):
    """
    Create a response from the model by using a formatted prompt
    """
    input_ids = tokenizer(
        PROMPT.format(instruction=instruction), return_tensors="pt"
    ).input_ids

    if device:
        input_ids = input_ids.to(device)
        model = model.to(device)

    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )
    decoded = tokenizer.batch_decode(gen_tokens)[0]

    # The response appears after "### Response:".  The model has been trained to append "### End" at the end.
    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", decoded, flags=re.DOTALL)

    response = None
    if m:
        response = m.group(1).strip()
    else:
        # The model might not generate the "### End" sequence before reaching the max tokens.  In this case, return
        # everything after "### Response:".
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            response = m.group(1).strip()
        else:
            pass
    return response

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        res_tok_id = self.tokenizer.encode(RESPONSE_KEY)
        labels = batch['labels'].clone()

        for i in range(len(examples)):
            res_tok_id_start_idx = None
            for idx in np.where(batch['labels'][i] == res_tok_id[0])[0]:
                res_tok_id_start_idx = idx
                break

            if res_tok_id_start_idx:
                labels[i, :res_tok_id_start_idx + 1] = -100

        batch['labels'] = labels

        return batch
    
def get_model_and_tokenizer(model_id = MODEL_ID, gradient_checkpointing = False):
    """
    Get the pretrained model and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code = True, use_cache = False if gradient_checkpointing else True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens' : [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY]})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def preprocess_batch(batch, tokenizer, max_length):
    """
    Preprocess a batch of the dataset
    """
    return tokenizer(
        batch['text'],
        max_length = max_length,
        truncation = True
    )

def preprocess_dataset(tokenizer, max_length, dataset_name = DATASET, seed = SEED):
    """
    Preprocess the entire dataset
    """
    dataset = load_dataset(dataset_name)['train']

    # Create the 'text' column
    dataset = dataset.to_pandas()

    def create_full_text(row):
        instruction = row.instruction
        if row.context:
            prompt = PROMPT_WITH_INPUT.format(instruction = instruction, input = row.context)
        else:
            prompt = PROMPT.format(instruction = instruction)
        prompt += row.response
        prompt += f'\n\n{END_KEY}'
        return prompt
    
    dataset['text'] = dataset.apply(create_full_text, axis = 1)
    dataset = Dataset.from_pandas(dataset)

    for i in range(5):
        print(dataset['text'][i])
        print('\n\n')
        print('-'*20)
        print('\n\n')


    dataset = dataset.filter(lambda rec : not rec['text'].strip().endswith(RESPONSE_KEY.strip()))

    _preproc_func = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    dataset = dataset.map(
        _preproc_func,
        batched = True,
        remove_columns = ['instruction', 'context', 'response', 'category', 'text']
    )

    dataset = dataset.shuffle(seed = seed)
    return dataset

def train(
        local_output_dir,
        epochs,
        train_batch_size,
        eval_batch_size,
        lr,
        seed,
        gradient_checkpointing,
        cuda,
        test_size = 1000,
        model_id = MODEL_ID,
        fsdp = True
):
    """
    Train DLite
    """
    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_id = model_id, gradient_checkpointing = gradient_checkpointing)
    conf = model.config
    max_length = getattr(conf, 'n_positions', getattr(conf, 'seq_length', 1024))

    processed_dataset = preprocess_dataset(tokenizer, max_length)
    split_dataset = processed_dataset.train_test_split(test_size = test_size, seed = seed)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer = tokenizer,
        mlm = False,
        return_tensors = 'pt',
        pad_to_multiple_of = 8
    )

    training_args = TrainingArguments(
        output_dir = local_output_dir,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        learning_rate = lr,
        num_train_epochs = epochs,
        gradient_checkpointing = gradient_checkpointing,
        logging_dir = f'{local_output_dir}/runs',
        logging_strategy = 'steps',
        logging_steps = 10,
        evaluation_strategy = 'steps',
        eval_steps = 100,
        save_strategy = 'steps',
        save_steps = 200,
        save_total_limit = 1,
        load_best_model_at_end = True,
        report_to = 'tensorboard',
        disable_tqdm = True,
        remove_unused_columns = False,
        no_cuda = not cuda,
        fsdp = fsdp
    )

    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = split_dataset['train'],
        eval_dataset = split_dataset['test'],
        data_collator = data_collator
    )
    trainer.train()

    trainer.save_model(local_output_dir)
