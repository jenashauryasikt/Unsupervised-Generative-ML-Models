## Import required libraries
from typing import Callable, Iterator, Sequence
import functools

import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
import matplotlib.pylab as plt

import jax.numpy as jnp
import jax.scipy as jsp
import jax
import flax.linen as nn
import optax
import haiku as hk
import chex
import tqdm

'''
All MLP codes take heavy inspiration from the starter code
provided in the score matching notebook.
'''
Activation = Callable[[jax.Array], jax.Array]

## MLP for EBM
class EBM_MLP(nn.Module):    
    features: Sequence[int]
    activation: Activation = nn.swish
    out_activation: Activation = nn.sigmoid

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.features[:-1]:
            x = nn.Dense(f)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
    
## MLP for DSM
class DSM_MLP(nn.Module):
    
    features: Sequence[int]
    activation: Activation = nn.swish

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.features[:-1]:
            x = nn.Dense(f)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x
    
## MLP for DDPM
class DDPM_MLP(nn.Module):
    features: Sequence[int]
    num_steps: int
    activation: Activation = nn.swish

    @nn.compact
    def __call__(self, x: jax.Array, t: int) -> jax.Array:
        # time_embs = self.time_embeds(t, x)
        time_embs = nn.Embed(self.num_steps, self.features[0])(jnp.array([t]))
        time_embs = jnp.broadcast_to(time_embs, (x.shape[0], self.features[0]))
        x = jnp.concatenate([x, time_embs], axis=-1)
        for f in self.features[:-1]:
            x = nn.Dense(f)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x

## GAN MLPs
## Generator
class GAN_generator(nn.Module):

    features: Sequence[int]
    activation: Activation = nn.swish
    # running_avg: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.features[:-1]:
            x = nn.Dense(f)(x)
            # x = nn.BatchNorm(use_running_average=self.running_avg)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        return x

## Discriminator 
class GAN_discriminator(nn.Module):

    features: Sequence[int]
    activation: Activation = nn.swish
    # running_avg: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.features[:-1]:
            x = nn.Dense(f)(x)
            # x = nn.BatchNorm(use_running_average=self.running_avg)(x)
            x = self.activation(x)
        x = nn.Dense(self.features[-1])(x)
        x = nn.sigmoid(x) # probabilities between 0 and 1
        return x
    
## optimizer with gradient clipping to solve exploding gradients
def clipper_optimizer(learning_rate: float, clip_norm: float):
    gradient_transformation = optax.chain(
        optax.clip_by_global_norm(clip_norm),  # Clip gradients
        optax.adam(learning_rate)  # Adam optimization
    )
    return gradient_transformation

## Batch generator of dataset
'''
This code is taken directly from the starter code to divide the dataset into batches.
'''
class BatchManager(Iterator[np.ndarray]):
    
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        key: chex.PRNGKey
    ):
        batch_size = min(batch_size, len(data))
        self._num_batches = len(data) // batch_size
        self._batch_idx = None
        self._batch_size = batch_size
        self._key = hk.PRNGSequence(key)
        self._data = data
        self._reset()

    @property
    def num_batches(self) -> int:
        return self._num_batches
    
    def _reset(self) -> None:
        self._perm = np.array(jax.random.permutation(next(self._key), np.arange(len(self._data))))
        self._batch_idx = 0

    def __next__(self) -> np.ndarray:
        assert self._batch_idx is not None
        assert self._batch_idx >= 0 and self._batch_idx < self._num_batches
        inds = self._perm[self._batch_idx * self._batch_size : (self._batch_idx + 1) * self._batch_size]
        batch = self._data[inds]
        self._batch_idx += 1
        if self._batch_idx >= self._num_batches:
            self._reset()
        return batch
    
## Quantitative comparison via Gaussian KDE
def gen_logpdf(X, gen):
    gen_clean = gen[~np.isnan(gen).any(axis=-1)] #remove nan samples
    kde = gaussian_kde(X.T)
    logpdfs = kde.logpdf(gen_clean.T)
    print(f"No. of clean samples = {len(gen_clean)}")
    return logpdfs

## KDE Percentile plots
def plot_logpdf(X, gen, model_name, dataset_name):
    title = f'log(pdf) percentiles: {model_name} for {dataset_name}'
    logpdfs = gen_logpdf(X, gen)
    avg_logpdf = np.mean(logpdfs)
    print(f"Mean log(pdf) for {model_name} generated {dataset_name} dataset = {avg_logpdf}")
    sorted_logpdfs = np.sort(logpdfs)[::-1]
    percentiles = np.linspace(0, 100, len(sorted_logpdfs))
    p25 = np.percentile(logpdfs, 25)
    p50 = np.percentile(logpdfs, 50)
    p75 = np.percentile(logpdfs, 75)
    plt.figure()
    plt.scatter(percentiles, sorted_logpdfs)
    plt.xlabel('Percentile')
    plt.ylabel('Log PDF')

    plt.axhline(p25, color='r', linestyle='--', label=f'75th Percentile : {p25:.2f}')
    plt.axhline(p50, color='g', linestyle='--', label=f'50th Percentile : {p50:.2f}')
    plt.axhline(p75, color='b', linestyle='--', label=f'25th Percentile : {p75:.2f}')

    plt.legend()
    plt.title(title)
    plt.show()

'''
Energy plotting methods were closely developed following
my discussions with collaborator Eric Killian.
'''
   
## EBM Energy plotting function
def ebm_plot_energy_landscape(model, params, grid_x, grid_y):
    # meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)
    # flatten for model inputs
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    grid_points_jax = jnp.array(grid_points)

    energies = model.apply(params, grid_points_jax).reshape(X.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, energies, levels=100, cmap="viridis")
    plt.colorbar(label="Energy")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Energy Landscape")
    plt.show()

## DSM Energy plotting function
def dsm_plot_energy_landscape(model, params, grid_x, grid_y):
    # meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)
    # flatten for model inputs
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    grid_points_jax = jnp.array(grid_points)

    energies = model.apply(params, grid_points_jax)
    energies_norm = jnp.linalg.norm(energies, axis=-1).reshape(X.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, energies_norm, levels=100, cmap="viridis")
    plt.colorbar(label="Energy")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Energy Landscape")
    plt.show()

## DDPM Energy plotting function
def ddpm_plot_energy_landscape(model, params, grid_x, grid_y):
    # meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)
    # flatten for model inputs
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    grid_points_jax = jnp.array(grid_points)

    # t=1 for the final denoised generations
    t = 1
    energies = model.apply(params, grid_points_jax, t)
    energies_norm = jnp.linalg.norm(energies, axis=-1).reshape(X.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, energies_norm, levels=100, cmap="viridis")
    plt.colorbar(label="Energy")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Energy Landscape")
    plt.show()

## GAN plot energy landscape
def gan_plot_energy_landscape(model, params, grid_x, grid_y):
    # meshgrid
    X, Y = np.meshgrid(grid_x, grid_y)
    # flatten for model inputs
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    grid_points_jax = jnp.array(grid_points)
    
    energies = model.apply(params, grid_points_jax)
    energies_norm = jnp.linalg.norm(energies, axis=-1).reshape(X.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, energies_norm, levels=100, cmap="viridis")
    plt.colorbar(label="Energy")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Energy Landscape")
    plt.show()