"""LAD Loss/Gain/Same Fine-tuning Script

Fine-tunes a pretrained DNA sequence model for LAD classification with 4 classes:
- none (0)
- same (1)
- lost (2)
- gained (3)
"""

import argparse
import functools
import os
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from jax.profiler import trace
from tensorboardX import SummaryWriter

import data_shae
from modelling import model

def parse_args():
    parser = argparse.ArgumentParser(description="LAD Loss/Gain/Same Fine-tuning Script")
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension")
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier")
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--vocab_size", type=int, default=8, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--max_lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--log_every", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--data_dir", type=str, default="gs://minformer_data/lad_finetuning_tfrecords_test", 
                       help="Directory containing training TFRecord files")
    parser.add_argument("--test_data_dir", type=str, default="gs://minformer_data/lad_finetuning_tfrecords_test_set", 
                       help="Directory containing test TFRecord files")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/lad_finetuning", 
                       help="Base directory for TensorBoard logs")
    parser.add_argument("--pretrained_checkpoint_dir", type=str, 
                       default="gs://minformer_data/pretrained_ckpt/v1", 
                       help="Directory for loading pretrained checkpoints")
    parser.add_argument("--finetuned_checkpoint_dir", type=str, 
                       default="gs://minformer_data/lad_finetuning_ckpt", 
                       help="Directory for saving fine-tuned checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, 
                       help="Save checkpoint every N steps")
    return parser.parse_args()

def create_lad_iterator(tfrecord_files: str, batch_size: int, shuffle: bool = False):
    """Creates iterator for LAD fine-tuning data."""
    def feature_description():
        return {
            "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "labels": tf.io.FixedLenFeature([], tf.int64)
        }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description())

    files = tf.io.gfile.glob(tfrecord_files)
    print(f"Found {len(files)} files")
    
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function, num_parallel_calls=32)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(32)

    for batch in dataset:
        yield {
            "x": batch["x"].numpy().astype(np.int32),
            "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
            "y": batch["labels"].numpy().astype(np.int32)
        }

def compute_lad_loss(
    weights: Any,
    x: jax.Array,
    segment_ids: jax.Array,
    y: jax.Array,
    cfg: model.Config,
    aux: Any | None = None,
) -> tuple[jax.Array, Any]:
    """Compute loss for LAD fine-tuning."""
    logits, internals, _ = model.forward(x, segment_ids, weights, cfg, aux=aux)
    
    # Create mask for valid tokens
    loss_mask = jnp.where(segment_ids == 0, 0, 1)
    
    # Compute cross entropy loss and accuracy
    loss, accuracy = model.cross_entropy_loss(logits, y, loss_mask)
    internals['acc'] = accuracy
    internals['predictions'] = logits
    
    return loss, internals

def log_metrics(writer, metrics, step):
    """Log metrics to TensorBoard."""
    flat_metrics = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)

def main():
    args = parse_args()

    # Create unique log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"lad_d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

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
        total_steps=args.total_steps
    )

    # Print configuration
    print("\nModel Configuration:")
    for field in cfg.__dataclass_fields__:
        print(f"{field}: {getattr(cfg, field)}")

    # Set up checkpoint managers
    pretrained_ckpt_manager = model.make_mngr(path=args.pretrained_checkpoint_dir)
    finetuned_ckpt_manager = model.make_mngr(path=args.finetuned_checkpoint_dir)
    
    # Load pretrained weights
    print(f"\nLoading pretrained weights from: {args.pretrained_checkpoint_dir}")
    weights, opt_state = model.load(pretrained_ckpt_manager, cfg)
    
    # Initialize new layers for LAD prediction with proper sharding
    predictor_info = model.TensorInfo(
        jax.ShapeDtypeStruct((cfg.d_model, 4), cfg.weight_dtype_at_rest),
        ("d_model", "vocab"),
        jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
    )

    hidden_info = model.TensorInfo(
        jax.ShapeDtypeStruct((cfg.d_model, cfg.d_model), cfg.weight_dtype_at_rest),
        ("d_model", "ffw"),
        jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
    )

    # Initialize with proper sharding
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    hidden_mean = jax.device_put(
        hidden_info.initializer(key1, hidden_info.shape.shape, hidden_info.shape.dtype),
        model._logical_to_sharding(hidden_info.logical_axes, cfg.mesh, cfg.rules)
    )

    predictor_mean = jax.device_put(
        predictor_info.initializer(key2, predictor_info.shape.shape, predictor_info.shape.dtype),
        model._logical_to_sharding(predictor_info.logical_axes, cfg.mesh, cfg.rules)
    )

    # Combine pretrained weights with new layers
    weights_for_finetuning = (weights, hidden_mean, predictor_mean)
    opt_state_for_finetuning = (opt_state, 
                               model.init_optimizer_state(hidden_mean),
                               model.init_optimizer_state(predictor_mean))

    # Create data iterator
    train_iterator = create_lad_iterator(
        f"{args.data_dir}/record_*.tfrecord",
        batch_size=args.batch_size,
        shuffle=True
    )

    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames=["cfg", "override_compute_loss_fn"])
    step = functools.partial(step, cfg=cfg, override_compute_loss_fn=compute_lad_loss)

    # Training loop
    print("\nStarting fine-tuning...")
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

        for i in range(cfg.total_steps):
            # Get next batch
            batch = next(train_iterator)
            batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))

            # Profile first step
            if i == 0:
                with trace(log_dir):
                    loss, weights_for_finetuning, opt_state_for_finetuning, internals = step(
                        weights_for_finetuning,
                        batch["x"],
                        batch["segment_ids"],
                        batch["y"],
                        opt_state_for_finetuning,
                        i,
                    )
                    jax.block_until_ready(loss)
            else:
                loss, weights_for_finetuning, opt_state_for_finetuning, internals = step(
                    weights_for_finetuning,
                    batch["x"],
                    batch["segment_ids"],
                    batch["y"],
                    opt_state_for_finetuning,
                    i,
                )

            # Log metrics
            if i % args.log_every == 0:
                writer.add_scalar("loss", loss, i)
                writer.add_scalar("accuracy", internals["acc"], i)
                print(f"Step {i}, Loss: {loss:.4f}, Accuracy: {internals['acc']:.4f}")
                log_metrics(writer, internals, i)

            # Save checkpoint
            if i > 0 and i % args.checkpoint_interval == 0:
                print(f"\nSaving checkpoint at step {i}")
                model.save(
                    finetuned_ckpt_manager,
                    weights_for_finetuning,
                    opt_state_for_finetuning,
                    i
                )

    print("\nTraining completed!")
    print("TensorBoard logs saved in:", log_dir)

if __name__ == "__main__":
    main()