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
dsm_model = DSM_MLP(features=[128, 128, 2])
optimizer = clipper_optimizer(1e-3, 0.1)

## DSM loss
@jax.jit
def denoising_score_matching_loss(params: chex.ArrayTree, batch: jax.Array, key: chex.PRNGKey, sigma: float) -> float:
    ws = jax.random.normal(key, shape=batch.shape)
    xs = batch + sigma * ws

    fs = dsm_model.apply(params, xs)
    # loss = jnp.sum(jnp.square(fs + (ws / sigma))): scale by sigma for convenience as suggested
    loss = jnp.mean(jnp.square((fs * sigma) + ws))
    return loss

## batch update for learning
@jax.jit
def dsm_update(
    batch: jax.Array,
    params: chex.ArrayTree, 
    opt_state: optax.OptState,
    sigma: float,
    key: chex.PRNGKey
) -> tuple[float, chex.ArrayTree, optax.OptState]:
    loss, grad = jax.value_and_grad(denoising_score_matching_loss)(params, batch, key, sigma=sigma)
    updates, opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, opt_state

## Training code
def DSM_train(X, lr, clip_norm, sigma, num_epochs, batch_size, key):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X_copy = X[idxs]
    X_train = X_copy[:-500]
    X_val = X_copy[-500:]

    # Initialize the DSM model
    params = dsm_model.init(next(key), X_train[:1, ...])
    optimizer = clipper_optimizer(lr, clip_norm)
    opt_state = optimizer.init(params)
    bm = BatchManager(data=X_train, batch_size=batch_size, key=next(key))
    val_bm = BatchManager(data=X_val, batch_size=len(X_val), key=next(key))

    train_losses = []
    val_losses = []
    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        batch = next(bm)
        val_batch = next(val_bm)
        loss, params, opt_state = dsm_update(batch, params, opt_state, sigma, key=next(key))
        val_loss, _, _ = dsm_update(val_batch, params, opt_state, sigma, key=next(key))
        train_losses.append(loss)
        val_losses.append(val_loss)

    return params, train_losses, val_losses

## Loss plots
def DSM_training_plot(train_losses, val_losses, title):
    epochs = np.arange(0, len(train_losses))
    plt.figure()
    plt.plot(epochs, train_losses, color='b', label='Training')
    plt.plot(epochs, val_losses, color='r', label='Validation')
    plt.title(title)
    plt.legend()
    plt.show()

'''
Inspired from the given score matching notebook.
'''
## DSM Langevin Sampler
@functools.partial(jax.jit, static_argnames=("num_steps",))
def dsm_langevin_sampling(
    params: chex.ArrayTree,
    key: chex.PRNGKey,
    step_size: float,
    initial_samples: jax.Array,
    num_steps: int,
) -> jax.Array:

    def scan_fn(carry, _):
        states, key = carry
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, shape=states.shape)
        next_states = states + step_size * dsm_model.apply(params, states) + jnp.sqrt(2 * step_size) * noise
        return (next_states, key), None

    states = initial_samples
    (states, _), _ = jax.lax.scan(scan_fn, (states, key), jnp.arange(num_steps))
    return states

## Generation
def DSM_sampler(params, X, noise, step_size, num_steps, key):
    samples = dsm_langevin_sampling(
    params,
    next(key),
    step_size,
    noise,
    num_steps)

    plt.figure()
    plt.plot(samples[:, 0], samples[:, 1], '.', label='Generated')
    plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2, label='Original')
    plt.xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
    plt.ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
    plt.legend()
    plt.show()
    return samples