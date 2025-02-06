"""


For open genome:

python3 projects/bio/train.py --checkpoint_dir=/tmp/bio_checkpoints/test_run --checkpoint_interval=10000 --max_seq_len=16384 --data_dir=gs://minformer_data/open-genome-imgpr/tfrecords/stage1/train_v3/ --log_every=10
python3 projects/bio/train.py --checkpoint_dir=/tmp/bio_checkpoints/test_run_fp32norm --checkpoint_interval=1000 --max_seq_len=8192 --dataset=shae_8k --log_every=10

"""

import argparse
import functools
import os
import wandb
from datetime import datetime
from typing import Any

# Assuming these are in the same directory or in the Python path
import data
import data_hf
import data_shae
import jax
import jax.numpy as jnp
import modelling.model as model
import numpy as np
from jax.profiler import trace
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="DNA Sequence Training Script")
    # Model architecture parameters
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension") # Model dimension
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier") # Feed-forward network size multiplier
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads") # Number of attention heads
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads") 
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers") # Number of transformer layers
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--vocab_size", type=int, default=6, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate model every N steps")
    parser.add_argument("--data_dir", type=str, default="data/tfrecords/", help="Directory containing TFRecord files")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/shae", help="Base directory for TensorBoard logs")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/tmp/dna_checkpoints", help="Directory for saving checkpoints"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["human-genome-8192", "open-genome-imgpr", "shae_8k", "diverse_genomes"],
        default="diverse_genomes",
        help="Type of dataset to download and process",
    )
    return parser.parse_args()


def clean_key(key):
    cleaned = key.replace("['", "").replace("']", "")
    cleaned = cleaned.replace(".", "/")
    return cleaned


def flatten_pytree(tree):
    leaves = jax.tree_util.tree_map_with_path(lambda p, x: (clean_key(jax.tree_util.keystr(p)), x), tree)
    return jax.tree_util.tree_leaves(leaves, is_leaf=lambda x: isinstance(x, tuple))


def log_metrics(writer, metrics, step):
    flat_metrics = flatten_pytree(metrics)
    for key, value in flat_metrics:
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)


def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(
        project="dna-transformer",
        config={
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "query_heads": args.query_heads,
            "key_heads": args.key_heads,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "dataset": args.dataset
        }
    )

    # Create a unique log directory name with key configuration parameters
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

    # Data loading setup
    if args.dataset == "open-genome-imgpr":
        iter = data_hf.create_iterator(
            str(args.data_dir) + "record_*.tfrecord", batch_size=args.batch_size, shuffle=True
        )
        process_batch = model.process_batch
    elif args.dataset == "shae_8k":
        eukaroytes = ['drosophila_genome_8192bp_bins_no_N',
                    'macaque_genome_8192bp_bins_no_N',
                    'mouse_genome_8192bp_bins_no_N',
                    'zebrafish_genome_8192bp_bins_no_N',
                    '8kb_genomic_bins_with_sequences_GW17IPC']
        stage_1 = [f"gs://minformer_data/{e}/tfrecords/record_*.tfrecord" for e in eukaroytes]
        # Do second stage on human only.
        stage_2 = [ "gs://minformer_data/shae_8k/tfrecords/record_*.tfrecord"]
        iter = data_shae.create_iterator(
           stage_1=stage_1, stage_2=stage_2, batch_size=args.batch_size, shuffle=True
        )
        process_batch = model.process_batch_shae
    
    elif args.dataset == "diverse_genomes":
        species = [
            'Bradyrhizobium_japonicum_8192bp_bins_no_N',
            'Burkholderia_pseudomallei_8192bp_bins_no_N',
            'Caenorhabditis_elegans_8192bp_bins_no_N',
            'Combined_viruses_8192bp_bins_no_N',
            'Monodelphis_domestica_8192bp_bins_no_N',
            'Ornithorhynchus_anatinus_8192bp_bins_no_N',
            'Pseudomonas_fluorescens_8192bp_bins_no_N',
            'Rhodococcus_jostii_8192bp_bins_no_N',
            'bacteroides_genome_8192bp_bins_no_N',
            'borrelia_genome_8192bp_bins_no_N',
            'candida_genome_8192bp_bins_no_N',
            'chlamydia_genome_8192bp_bins_no_N',
            'drosophila_genome_8192bp_bins_no_N',
            'ecoli_genome_8192bp_bins_no_N',
            'haloferax_genome_8192bp_bins_no_N',
            'human_genome_8192bp_bins_no_N',
            'macaque_genome_8192bp_bins_no_N',
            'methanogen_genome_8192bp_bins_no_N',
            'mouse_genome_8192bp_bins_no_N',
            'myxococcus_genome_8192bp_bins_no_N',
            'pombe_genome_8192bp_bins_no_N',
            'pseudomonas_genome_8192bp_bins_no_N',
            'pylori_genome_8192bp_bins_no_N',
            'saccharomyces_cerevisiae_8192bp_bins_no_N',
            'streptomyces_genome_8192bp_bins_no_N',
            'subtilis_genome_8192bp_bins_no_N',
            'sulfolobus_genome_8192bp_bins_no_N',
            'tb_genome_8192bp_bins_no_N',
            'thermus_genome_8192bp_bins_no_N',
            'zebrafish_genome_8192bp_bins_no_N'
        ]
        file_patterns = [f"gs://minformer_data/diverse_genomes_tf_v2/{s}/tfrecords/record_*.tfrecord" for s in species]
        iter = data.DNADataset(sequence_length=8192).create_iterator(
            file_pattern=file_patterns,
            batch_size=args.batch_size,
            shuffle=True
        )
        process_batch = model.process_batch

    # Model configuration
    cfg = model.Config(
        d_model=args.d_model,
        ffw_multiplier=args.ffw_multiplier,
        query_heads=args.query_heads,
        key_heads=args.key_heads,
        num_layers=args.num_layers,
        key_dim=args.key_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        causal=True,
        use_attn_kernel=True,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.bfloat16,
        rules=model.fsdp_rules,
        mesh=model.create_mesh(),
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        # mega_byte=True,
        # patch_size=8,
    )

    # Print the config to verify all parameters are set correctly
    print("Model Configuration:")
    for field in cfg.__dataclass_fields__:
        print(f"{field}: {getattr(cfg, field)}")

    # Checkpoint manager setup
    ckpt_manager = model.make_mngr(path=args.checkpoint_dir)

    # Initialize or load weights and optimizer state
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        weights, opt_state = model.load(ckpt_manager, cfg)
        start_step = ckpt_manager.latest_step()
    else:
        print("Initializing new weights...")
        weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
        opt_state = model.init_optimizer_state(weights)
        start_step = 0

    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames="cfg")
    step = functools.partial(step, cfg=cfg)

    # Training loop
    with SummaryWriter(log_dir) as writer:
        # Log hyperparameters
        writer.add_hparams(
            {
                "d_model": cfg.d_model,
                "num_layers": cfg.num_layers,
                "query_heads": cfg.query_heads,
                "key_heads": cfg.key_heads,
                "max_lr": cfg.max_lr,
                "min_lr": cfg.min_lr,
                "warmup_steps": cfg.warmup_steps,
                "total_steps": cfg.total_steps,
                "batch_size": args.batch_size,
                "max_seq_len": cfg.max_seq_len,
            },
            {},
        )

        for i in range(start_step, cfg.total_steps):
            next_batch = next(iter) # Get next batch
            batch = process_batch(next_batch, cfg, step_idx=i)
            batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))

            # Always profile on the first step so that we can think about optimisations.
            if i == 0:
                with trace(log_dir):
                    loss, weights, opt_state, internals = step(
                        weights,
                        batch["x"],
                        batch["segment_ids"],
                        batch["y"],
                        opt_state,
                        i,
                        aux=batch["aux"],
                    )
                    jax.block_until_ready(loss)
            else:
                loss, weights, opt_state, internals = step(
                    weights, batch["x"], batch["segment_ids"], batch["y"], opt_state, i, aux=batch["aux"]
                )

            if i % args.log_every == 0:
                # Log loss and accuracy to TensorBoard
                writer.add_scalar("loss", loss, i)
                writer.add_scalar("accuracy", internals["accuracy"], i)
                writer.add_scalar("num_tokens_per_batch", np.sum(batch["segment_ids"] != 0), i)
                #print(f"Step {i}, Loss: {loss}, Accuracy: {internals['accuracy']}")
                log_metrics(writer, internals, i)

                # Log to wandb
                metrics = {
                    "loss": loss,
                    "accuracy": internals["accuracy"],
                    "num_tokens_per_batch": np.sum(batch["segment_ids"] !=0),
                    "learning_rate": internals["lr"],
                    "step": i,
                }

                # Log grad norms if available
                if "grad_norms" in internals:
                    grad_norms = jax.tree_util.tree_leaves(internals["grad_norms"])
                    metrics["grad_norm_avg"] = np.mean([x.item() for x in grad_norms])
                
                wandb.log(metrics)
                print(f"Step {i}, Loss: {loss}, Accuracy: {internals['accuracy']}")

            if i % 1000 == 0:  # Every 100 steps
                # First run a forward pass to get logits
                logits, internals, _ = model.forward(
                    batch["x"], 
                    batch["segment_ids"], 
                    weights, 
                    cfg, 
                    cache=None
                )
                
                # Convert to numpy and then to Python integers
                print("\nToken distributions:")
                pred_logits = np.array(logits[0])  # Convert to numpy
                pred_tokens = [int(x) for x in np.argmax(pred_logits, axis=-1)]  # Convert to Python ints
                true_tokens = [int(x) for x in np.array(batch["y"][0])]  # Convert to Python ints

                print("Predicted token counts:", np.unique(pred_tokens, return_counts=True))
                print("True token counts:", np.unique(true_tokens, return_counts=True))

                # Create DNA dataset instance for decoding
                dna_dataset = data.DNADataset()

                # Decode both sequences
                pred_seq = dna_dataset.detokenize(pred_tokens)
                true_seq = dna_dataset.detokenize(true_tokens)

                # Print first 50 bases to keep output readable
                print("\nExample Prediction:")
                print(f"Predicted: {pred_seq[:100]}...")
                print(f"Actual:    {true_seq[:100]}...")
                print("-" * 60)


            # Save checkpoint
            if i > 0 and i % args.checkpoint_interval == 0:
                print(f"Saving checkpoint at step {i}")
                model.save(ckpt_manager, weights, opt_state, i)

    wandb.finish()
    print("Training completed. TensorBoard logs saved in:", log_dir)


if __name__ == "__main__":
    main()
