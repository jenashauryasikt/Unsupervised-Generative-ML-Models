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
# The last layer has a single output for energy
ebm_model = EBM_MLP(features=[64, 64, 1], out_activation=None)
step_size = 1e-2
num_steps = 500
optimizer = clipper_optimizer(1e-3, 0.1)

## Energy function represented by the EBM MLP
@jax.jit
def energy_fn(params: chex.ArrayTree, inputs: jax.Array) -> jax.Array:
    # mlp_model = EBM_MLP(features=[64, 64, 2], out_activation=None)  
    energies = ebm_model.apply(params, inputs)
    return energies

## Langevin Sampling
'''
This code is taken from the starter code and altered for EBM.
'''
@functools.partial(jax.jit, static_argnames=("num_steps",))
def ebm_langevin_sampling(
    params: chex.ArrayTree,
    key: chex.PRNGKey,
    step_size: float,
    initial_samples: jax.Array,
    num_steps: int,
) -> jax.Array:
    
    def simple_energy_fn(x):
        return jnp.mean(ebm_model.apply(params, x))
    
    grad_energy = jax.jit(jax.grad(simple_energy_fn))

    def scan_fn(carry, _):
        states, key = carry
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, shape=states.shape)

        energy_grads = grad_energy(states)

        # next_states = states + step_size * ebm_model.apply(params, states) + jnp.sqrt(2 * step_size) * noise
        next_states = states - step_size * energy_grads + jnp.sqrt(2 * step_size) * noise
        return (next_states, key), None

    states = initial_samples
    # (states, _), _ = jax.lax.scan(scan_fn, (states, key), None, jnp.arange(num_steps))
    (states, _), _ = jax.lax.scan(scan_fn, (states, key), None, length=num_steps)
    return states

# EBM loss function which aims to minimize the energy of the samples
@jax.jit
def ebm_loss_fn(params: chex.ArrayTree, inputs: jax.Array, alpha: float, key: chex.PRNGKey) -> float:
    samples = ebm_langevin_sampling(
        params,
        key,
        step_size,
        2 * jax.random.normal(key, shape=inputs.shape),
        num_steps)
    neg_energy = energy_fn(params, samples)
    pos_energy = energy_fn(params, inputs)

    # regularization
    l2_penalty = alpha * (jnp.mean(jnp.square(pos_energy)) + jnp.mean(jnp.square(neg_energy)))
    loss = jnp.mean(pos_energy) - jnp.mean(neg_energy) + l2_penalty  # Mean energy loss
    return loss

## batch update for learning
@jax.jit
def ebm_update(params: chex.ArrayTree, batch: jax.Array, opt_state: optax.OptState, alpha: float, key: chex.PRNGKey) -> tuple[float, chex.ArrayTree, optax.OptState]:
    """perform a single update step on a batch of data."""
    loss, grads = jax.value_and_grad(ebm_loss_fn)(params, batch, alpha, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, opt_state

## Training code
def EBM_train(X, lr, clip_norm, l2, num_epochs, batch_size, key):
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X_copy = X[idxs]
    X_train = X_copy[:-500]
    X_val = X_copy[-500:]

    # Initialize the EBM model
    params = ebm_model.init(next(key), X_train[:1, ...])
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
        loss, params, opt_state = ebm_update(params, batch, opt_state, l2, key=next(key))
        # val_loss = ebm_loss_fn(params, X_val, l2, next(key))
        val_loss, _, _ = ebm_update(params, val_batch, opt_state, l2, key=next(key))
        train_losses.append(loss)
        val_losses.append(val_loss)

    return params, train_losses, val_losses

## Loss plots
def EBM_training_plot(train_losses, val_losses, title):
    epochs = np.arange(0, len(train_losses))
    plt.figure()
    plt.plot(epochs, train_losses, color='b', label='Training')
    plt.plot(epochs, val_losses, color='r', label='Validation')
    plt.title(title)
    plt.legend()
    plt.show()

## Generation
def EBM_sampler(params, X, noise, step_size, num_steps, key):
    samples = ebm_langevin_sampling(
    params,
    next(key),
    step_size,
    noise,
    num_steps)

    plt.figure()
    plt.plot(samples[:, 0], samples[:, 1], '.', label='Generated')
    plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2, label='Original')
    plt.xlim(np.min(X[:, 0])-1, np.max(X[:, 0])+1)
    plt.ylim(np.min(X[:, 1])-1, np.max(X[:, 1])+1)
    plt.legend()
    plt.show()
    return samples