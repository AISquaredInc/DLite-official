from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from datasets import load_dataset, Dataset
from functools import partial
from tqdm import tqdm
import numpy as np
import re

DATASET = 'aisquared/databricks-dolly-15k'
MODEL_ID = 'gpt2'
END_KEY = '### End'
INSTRUCTION_KEY = '### Instruction:'
RESPONSE_KEY = '### Response:\n'
SEED = 42
DEFAULT_MAX_LENGTH = 1024
PROMPT_SEED = 'The following is an instruction that describes a task, along with any additional context. Write a response that appropriately completes the request.'

PROMPT_WITH_INPUT = """%s

%s
{instruction}

{input}

%s""" % (PROMPT_SEED, INSTRUCTION_KEY, RESPONSE_KEY)

PROMPT = """%s

%s
{instruction}

%s""" % (PROMPT_SEED, INSTRUCTION_KEY, RESPONSE_KEY)

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
        trust_remote_code = True,
        use_fast = False
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast = False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens' : [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY]})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def preprocess_batch(batch, tokenizer, max_length = DEFAULT_MAX_LENGTH):
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

    _preproc_func = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)

    # If training off of the dolly-15k dataset
    if dataset_name == 'aisquared/databricks-dolly-15k':
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

        # Filter out prompts that are too long
        text_lengths = dataset.text.apply(lambda s : len(tokenizer(s)['input_ids']))
        dataset = dataset[text_lengths <= max_length].reset_index(drop = True)

        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.map(
            _preproc_func,
            batched = True,
            remove_columns = ['instruction', 'context', 'response', 'category', 'text']
        )
    
    # If training off of the alpaca dataset
    elif dataset_name == 'tatsu-lab/alpaca':
        dataset = dataset.filter(lambda rec : not rec['text'].strip().endswith(RESPONSE_KEY.strip()))

        def _func(rec):
            rec['text'] += f'\n\n{END_KEY}'
            return rec
        
        dataset = dataset.map(_func)
        dataset = dataset.map(
            _preproc_func,
            batched = True,
            remove_columns = ['instruction', 'input', 'output', 'text']
        )

    else:
        raise ValueError(f'Got unsupported dataset: {dataset_name}')

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
        deepspeed,
        test_size = 1000,
        model_id = MODEL_ID,
        local_rank = None,
        fp16 = False,
        max_length = DEFAULT_MAX_LENGTH,
        dataset = DATASET,
        load_best = True
):
    """
    Train DLite
    """
    set_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_id = model_id, gradient_checkpointing = gradient_checkpointing)
    conf = model.config
    max_length = getattr(conf, 'n_positions', getattr(conf, 'seq_length', max_length))

    processed_dataset = preprocess_dataset(tokenizer, max_length, dataset_name = dataset)
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
        save_total_limit = None,
        load_best_model_at_end = load_best,
        report_to = 'tensorboard',
        disable_tqdm = False,
        remove_unused_columns = False,
        no_cuda = not cuda,
        deepspeed = deepspeed,
        local_rank = local_rank,
        fp16 = fp16
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
