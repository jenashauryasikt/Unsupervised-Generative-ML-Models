## Import required libraries
from typing import Callable, Iterator, Sequence
import functools

import numpy as np
import scipy as sp
import matplotlib.pylab as plt

import jax.numpy as jnp
import jax.scipy as jsp
import jax
import flax.linen as nn
import optax
import haiku as hk
import chex
import tqdm

## Import helper codes
from utils import *

## Global variables
generator = GAN_generator(features=[256, 256, 2])
discriminator = GAN_discriminator(features=[256, 256, 1])
gen_optimizer = clipper_optimizer(1e-4, 0.1)
disc_optimizer = clipper_optimizer(1e-4, 0.1)

## generator loss
@jax.jit
def gen_loss_fn(
    gen_params: chex.ArrayTree, 
    disc_params: chex.ArrayTree, 
    noise: jax.Array, 
    eps: float
) -> float:
    samples = generator.apply(gen_params, noise)
    sample_probs = discriminator.apply(disc_params, samples)
    return -jnp.mean(jnp.log(sample_probs + eps))

## discriminator loss
@jax.jit
def disc_loss_fn(
    disc_params: chex.ArrayTree,
    inputs: jax.Array,
    samples: jax.Array,
    eps: float
) -> float:
    input_probs = discriminator.apply(disc_params, inputs)
    sample_probs = discriminator.apply(disc_params, samples)
    loss_input = -jnp.mean(jnp.log(input_probs + eps))
    loss_samples = -jnp.mean(jnp.log(1 - sample_probs + eps))
    return loss_input + loss_samples

## batch update
@jax.jit
def gan_update(
    batch: jax.Array,
    gen_params: chex.ArrayTree,
    disc_params: chex.ArrayTree,
    gen_opt_state: optax.OptState,
    disc_opt_state: optax.OptState,
    key: chex.PRNGKey,
    eps: float
) -> tuple[float, float, chex.ArrayTree, chex.ArrayTree, optax.OptState, optax.OptState]:
    # initial noise for sampling
    noise = 2 * jax.random.normal(key, batch.shape)
    # generator update
    gen_loss, gen_grads = jax.value_and_grad(gen_loss_fn)(gen_params, disc_params, noise, eps)
    gen_updates, new_gen_opt_state = gen_optimizer.update(gen_grads, gen_opt_state)
    new_gen_params = optax.apply_updates(gen_params, gen_updates)

    samples = generator.apply(new_gen_params, noise)
    #discriminator update
    disc_loss, disc_grads = jax.value_and_grad(disc_loss_fn)(disc_params, batch, samples, eps)
    disc_updates, new_disc_opt_state = disc_optimizer.update(disc_grads, disc_opt_state)
    new_disc_params = optax.apply_updates(disc_params, disc_updates)

    return gen_loss, disc_loss, new_gen_params, new_disc_params, new_gen_opt_state, new_disc_opt_state

## Training code
def GAN_train(X, lr, clip_norm, eps, num_epochs, batch_size, key):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X_copy = X[idxs]
    X_train = X_copy[:-500]
    X_val = X_copy[-500:]

    # Initialize the GAN
    gen_params = generator.init(next(key), X_train[:1, ...])
    disc_params = discriminator.init(next(key), X_train[:1, ...])
    gen_optimizer = clipper_optimizer(lr, clip_norm)
    disc_optimizer = clipper_optimizer(lr, clip_norm)
    gen_opt_state = gen_optimizer.init(gen_params)
    disc_opt_state = disc_optimizer.init(disc_params)
    bm = BatchManager(data=X_train, batch_size=batch_size, key=next(key))
    val_bm = BatchManager(data=X_val, batch_size=len(X_val), key=next(key))

    train_gen_losses, train_disc_losses = [], []
    val_gen_losses, val_disc_losses = [], []
    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        batch = next(bm)
        val_batch = next(val_bm)
        gen_loss, disc_loss, gen_params, disc_params, gen_opt_state, disc_opt_state = gan_update(
            batch, gen_params, disc_params, gen_opt_state, disc_opt_state, next(key), eps
        )
        val_gen_loss, val_disc_loss, _, _, _, _ = gan_update(
            val_batch, gen_params, disc_params, gen_opt_state, disc_opt_state, next(key), eps
        )
        train_gen_losses.append(gen_loss)
        train_disc_losses.append(disc_loss)
        val_gen_losses.append(val_gen_loss)
        val_disc_losses.append(val_disc_loss)

    return gen_params, train_gen_losses, train_disc_losses, val_gen_losses, val_disc_losses

## Loss plots
def GAN_training_plot(train_gen_losses, train_disc_losses, val_gen_losses, val_disc_losses, title):
    epochs = np.arange(0, len(train_gen_losses))
    plt.figure()
    plt.plot(epochs, train_gen_losses, color='b', label='Training (gen)')
    plt.plot(epochs, val_gen_losses, color='r', label='Validation (gen)')
    plt.title('Generator-'+title)
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(epochs, train_disc_losses, color='y', label='Training (disc)')
    plt.plot(epochs, val_disc_losses, color='g', label='Validation (disc)')
    plt.title('Discriminator'+title)
    plt.legend()
    plt.show()

## Generation
def GAN_sampler(gen_params, X, noise):
    samples = generator.apply(gen_params, noise)

    plt.figure()
    plt.plot(samples[:, 0], samples[:, 1], '.', label='Generated')
    plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2, label='Original')
    plt.xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
    plt.ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
    plt.legend()
    plt.show()
    return samples