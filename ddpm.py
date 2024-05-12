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
T = 1000
ddpm_model = DDPM_MLP(features=[128, 128, 2], num_steps=T)
optimizer = clipper_optimizer(learning_rate=1e-4, clip_norm=0.01)

## create noisy batch with diffusion
@jax.jit
def diffusion_process(x: jax.Array, alphas_cumprod: jax.Array, t: int, key: chex.PRNGKey) -> jax.Array:

    noise = jax.random.normal(key, x.shape)

    alpha_t = alphas_cumprod[t]
    xt = jnp.sqrt(alpha_t) * x + jnp.sqrt(1.0 - alpha_t) * noise
    return xt

## DDPM ELBO loss
@jax.jit
def ddpm_loss(params: chex.ArrayTree, x_batch: jax.Array, alphas_cumprod: jax.Array, t: int, key: chex.PRNGKey) -> float:
    
    ws = jax.random.normal(key, shape=x_batch.shape)
    
    xt = diffusion_process(x_batch, alphas_cumprod, t, key)
    fs = ddpm_model.apply(params, xt, t)
    loss = jnp.mean(jnp.square(fs - ws)) 
    return loss

## batch update for learning
@jax.jit
def ddpm_update(
    params: chex.ArrayTree,
    x_batch: jax.Array,
    alphas_cumprod: jax.Array,
    # alpha_bar_t: float,
    t: int,
    opt_state: optax.OptState,
    key: chex.PRNGKey
) -> tuple[float, chex.ArrayTree, optax.OptState]:
    loss, grad = jax.value_and_grad(ddpm_loss)(params, x_batch, alphas_cumprod, t, key)
    updates, opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, opt_state

## Training code
def DDPM_train(X, lr, clip_norm, sub_alpha_1, sub_alpha_T, T, num_epochs, batch_size, key):
    '''
    sub_alpha_1 and sub_alpha_T are 1 - alpha for ease of writing alphas.
    '''
    # Compute the noise schedule
    alphas_ = jnp.linspace(sub_alpha_1, sub_alpha_T, T)
    alphas_ = jnp.insert(alphas_, 0, 0)

    # Compute the alpha values
    alphas = 1.0 - alphas_
    alphas_cumprod = jnp.cumprod(alphas, axis=0)

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X_copy = X[idxs]
    X_train = X_copy[:-500]
    X_val = X_copy[-500:]

    # Initialize the DDPM model
    params = ddpm_model.init(next(key), X_train[:1, ...], 0)
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
        # randomly sample one t
        t = jax.random.randint(minval=1, maxval=T, key=next(key), shape=(1,))[0]
        loss, params, opt_state = ddpm_update(params, batch, alphas_cumprod, t, opt_state, jax.random.PRNGKey(t))
        val_loss, _, _ = ddpm_update(params, val_batch, alphas_cumprod, t, opt_state, jax.random.PRNGKey(t))
        train_losses.append(loss)
        val_losses.append(val_loss)

    return params, train_losses, val_losses, alphas, alphas_cumprod

## Loss plots
def DDPM_training_plot(train_losses, val_losses, title):
    epochs = np.arange(0, len(train_losses))
    plt.figure()
    plt.plot(epochs, train_losses, color='b', label='Training')
    plt.plot(epochs, val_losses, color='r', label='Validation')
    plt.title(title)
    plt.legend()
    plt.show()

## DDPM Sampler
@jax.jit
def ddpm_sampling(params: chex.ArrayTree, samples: jax.Array, alphas: jax.Array, alphas_cumprod: jax.Array) -> jax.Array:
    def scan_fn(x_t, t):
        key = jax.random.PRNGKey(t) # w_t for time t

        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_t_1 = alphas_cumprod[t-1]

        w_t = jax.random.normal(key, x_t.shape)
        f_t = ddpm_model.apply(params, x_t, t)

        x_t_1 = (1 / jnp.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t)/(jnp.sqrt(1 - alpha_bar_t))) * f_t
        ) + jnp.sqrt(((1 - alpha_t)*(1 - alpha_bar_t_1))/(1 - alpha_bar_t)) * w_t

        return x_t_1, x_t_1
    
    timesteps = jnp.arange(T-1, 1, -1)
    new_samples, _ = jax.lax.scan(scan_fn, samples, timesteps)

    return new_samples

## Generation
def DDPM_sampler(params, X, noise, alphas, alphas_cumprod):
    samples = ddpm_sampling(
    params,
    noise,
    alphas,
    alphas_cumprod)

    plt.figure()
    plt.plot(samples[:, 0], samples[:, 1], '.', label='Generated')
    plt.plot(X[:, 0], X[:, 1], 'o', alpha=0.2, label='Original')
    plt.xlim([np.min(X[:, 0])-1, np.max(X[:, 0])+1])
    plt.ylim([np.min(X[:, 1])-1, np.max(X[:, 1])+1])
    plt.legend()
    plt.show()
    return samples