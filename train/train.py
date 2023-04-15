from utils import train, SEED, MODEL_ID
import click

@click.command()
@click.argument('local-output-dir', type = click.Path(exists = False, dir_okay = True, file_okay = False))
@click.option('--epochs', '-e', type = int, default = 3)
@click.option('--train-batch-size', type = int, default = 16)
@click.option('--eval-batch-size', type = int, default = 16)
@click.option('--lr', type = float, default = 1e-5)
@click.option('--seed', type = int, default = SEED)
@click.option('--gradient-checkpointing/--no-gradient-checkpointing', default = True)
@click.option('--cuda/--no-cuda', default = True)
@click.option('--model-id', '-m', type = str, default = MODEL_ID)
@click.option('--deepspeed', type = click.Path(exists = True, file_okay = True, dir_okay = False), default = None)
@click.option('--local_rank', default = True)
@click.option('--fp16/--no-fp16', default = False)
def main(local_output_dir, epochs, train_batch_size, eval_batch_size, lr, seed, gradient_checkpointing, cuda, model_id, deepspeed, local_rank, fp16):
    train(
        local_output_dir = local_output_dir,
        epochs = epochs,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        lr = lr,
        seed = seed,
        gradient_checkpointing = gradient_checkpointing,
        cuda = cuda,
        model_id = model_id,
        deepspeed = deepspeed,
        local_rank = local_rank,
        fp16 = fp16
    )

if __name__ == '__main__':
    main()
