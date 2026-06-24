"""Procedural scene generation with realistic per-pixel spectral variance.

Increment B showed that flat solid-region scenes are *degenerate* — every pixel in a region
shares the identical spectrum, so the cube holds only a couple of unique spectra and the
perturbation-AE selection has nothing spatial to latch onto. These helpers paint each material
with a smooth random concentration field so spectra vary continuously across the scene (and
overlapping fields produce genuine mixtures), which is what real tissue looks like.
"""
from __future__ import annotations

import numpy as np

from spectraforge.scene import Scene


def random_field(height: int, width: int, seed: int, blur: int = 2) -> np.ndarray:
    """A smooth random concentration field in [0, 1] of shape (height, width), deterministic by seed."""
    rng = np.random.default_rng(seed)
    bh, bw = -(-height // 4), -(-width // 4)                 # ceil-div: coarse grid upsampled x4
    field = np.kron(rng.random((bh, bw)), np.ones((4, 4)))[:height, :width]
    for _ in range(blur):                                    # cheap box-ish blur -> smooth gradients
        field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                 + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
    span = float(field.max() - field.min())
    return (field - field.min()) / span if span else field


def random_scene(materials, height: int, width: int, seed: int) -> Scene:
    """Paint each material with its own smooth random concentration field (overlap -> mixtures)."""
    scene = Scene(height, width)
    for i, material in enumerate(materials):
        scene.paint_map(material, random_field(height, width, seed * 101 + i * 7))
    return scene


def make_confounded_scene(disc_materials, nuisance_materials, height: int, width: int, seed: int,
                          *, disc_amp: float = 1.0, nuisance_amp: float = 1.0, turbidity_amp: float = 1.0):
    """A LABELLED scene where band variance is deliberately DECOUPLED from informativeness.

    - ``disc_materials`` (the *discriminative* components — meant to be dim and/or spectrally
      overlapping) define the per-pixel class label (argmax of their i.i.d. concentration fields).
    - ``nuisance_materials`` (meant to be *bright* autofluorophores) get their own independent
      concentration fields — large spatial variance but **class-irrelevant**.
    - a ``scatter_field`` (turbidity) is returned to drive spatially-varying Rayleigh/Raman scatter —
      also high-variance, class-irrelevant.

    So the highest-variance bands (bright nuisances + scatter) carry no class information, while the
    discriminative signal sits in dim/overlapping (low-variance) bands. Returns
    ``(Scene, labels, scatter_field)``.
    """
    disc_fields = np.stack([random_field(height, width, seed * 13 + 31 * k)
                            for k in range(len(disc_materials))])
    labels = disc_fields.argmax(axis=0).astype(int)
    scene = Scene(height, width)
    for k, material in enumerate(disc_materials):
        scene.paint_map(material, disc_fields[k] * disc_amp)
    for j, material in enumerate(nuisance_materials):                       # class-irrelevant, own seeds
        field = random_field(height, width, seed * 271 + 17 * j + 9999)
        scene.paint_map(material, field * nuisance_amp)
    scatter_field = random_field(height, width, seed * 733 + 4242) * turbidity_amp
    return scene, labels, scatter_field


def make_interaction_scene(donor_material, acceptor_material, nuisance_materials, height: int,
                           width: int, seed: int, *, disc_amp: float = 1.0, nuisance_amp: float = 1.0,
                           turbidity_amp: float = 1.0):
    """A scene whose class label lives ENTIRELY in the *interaction* (co-localization) of two dyes.

    Two independent smooth fields drive a donor and an acceptor dye. Each is binarized at its median
    (b1, b2); the per-pixel class is ``b1 XOR b2`` — so each dye's MARGINAL concentration is
    class-independent (a linear/single-band test sees nothing) while the JOINT determines the class.
    Combined with renderer FRET (donor↔acceptor), the discriminative signal is relocated into the
    donor/acceptor band *ratio* via a nonlinear (saturating, product) coupling: a regime with a large
    *nonlinear-only* information fraction, where a linear selector (PCA) is at a fundamental
    disadvantage. Returns ``(Scene, labels, scatter_field)`` like :func:`make_confounded_scene`.
    """
    f1 = random_field(height, width, seed * 13 + 31)
    f2 = random_field(height, width, seed * 13 + 67)
    b1 = f1 > np.median(f1)
    b2 = f2 > np.median(f2)
    labels = (b1 ^ b2).astype(int)
    scene = Scene(height, width)
    scene.paint_map(donor_material, f1 * disc_amp)
    scene.paint_map(acceptor_material, f2 * disc_amp)
    for j, material in enumerate(nuisance_materials):
        field = random_field(height, width, seed * 271 + 17 * j + 9999)
        scene.paint_map(material, field * nuisance_amp)
    scatter_field = random_field(height, width, seed * 733 + 4242) * turbidity_amp
    return scene, labels, scatter_field


def make_labeled_scene(materials, height: int, width: int, seed: int):
    """A LABELLED, balanced scene for classification experiments.

    Each material gets its own smooth concentration field; the per-pixel class label is the
    argmax material (the dominant one). Because the fields are i.i.d. the classes come out
    balanced, the regions are organic (not blocks), and every pixel is a genuine mixture of all
    materials (heavier mixing near class boundaries) — realistic for material classification.

    Returns ``(Scene, labels)`` where ``labels`` is an (H, W) int array of class indices.
    """
    fields = np.stack([random_field(height, width, seed * 13 + 31 * k) for k in range(len(materials))])
    labels = fields.argmax(axis=0).astype(int)
    scene = Scene(height, width)
    for k, material in enumerate(materials):
        scene.paint_map(material, fields[k])
    return scene, labels
