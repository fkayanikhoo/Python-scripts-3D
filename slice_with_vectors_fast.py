#!/usr/bin/env python3

import numpy as np
from simext3d import *
from matplotlib import pyplot as plt
from my_14_numbers import *
import cmasher as cmr
import sys
import pandas
import my_matplotlib_style
from my_14_numbers import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import time
import os
import argparse


# =================Main function================
def main(args):

    print(f"started processing: {args.d}")


    sim = simext3d(args.d, tracer=True)

    print(f"read complete: {args.d}")


    sim.sort_scalar2grid("r")
    sim.sort_scalar2grid("th")
    sim.sort_scalar2grid("ph")
    sim.sort_scalar2grid("rho")
    sim.sort_scalar2grid("bsq")
    sim.sort_scalar2grid("Ehat")
    sim.sort_scalar2grid("tracer")
    sim.sort_scalar2grid("uint")
    sim.sort_vector2grid("ucon")
    sim.sort_vector2grid("bcon")
    sim.sort_tensor2grid("Rmunu")

    print(f"2grid complete: {args.d}")

    resolution = args.resolution
    span = args.span
    ugrid = np.linspace(-span, span, resolution)
    vgrid = np.linspace(-span, span, resolution)

    # Grids
    r = sim.r_grid
    theta = sim.th_grid
    phi = sim.ph_grid

    sim.rho_s_grid = sim.rho_grid * rhocgs
    sim.log_rho_s_grid = np.log10(sim.rho_s_grid)
    sim.log_Ehat_s_grid = np.log10(sim.Ehat_grid * endencgs)
    sim.uint_s_grid = sim.uint_grid * endencgs
    sim.log_uint_s_grid = np.log10(sim.uint_s_grid)

    # Parse scalar, vector, and slice arguments
    scalar_names = args.scalar
    vector_names = args.vector
    slice_names = [s.upper() for s in args.slice]

    # Axis vectors for slice setup
    axis_vectors = {"X": [0, 1, 0], "Y": [1, 0, 0], "Z": [0, 0, 1]}

    ugrid = np.linspace(-args.span, args.span, args.resolution)
    vgrid = np.linspace(-args.span, args.span, args.resolution)

    r_mask = r > 5

    # Plot configuration: name -> (label, vmin, vmax, cmap)
    plot_config = {
        "log_rho_s": (
            r"$\log_{10} \rho\,[\mathrm{g}\, \mathrm{cm}^{-3}]$",
            -8,
            -4,
            "turbo",
        ),
        "rho_s": (
            r"$\rho\,[\mathrm{g}\, \mathrm{cm}^{-3}]$",
            0,
            1e-4,
            "gnuplot",
        ),
        "uint_s": (
            r"$u_{int}\,[\mathrm{erg}\, \mathrm{cm}^{-3}]$",
            None,
            None,
            "magma",
        ),
        "log_uint_s": (
            r"$\log_{10}\, u_{int}\,[\mathrm{erg}\, \mathrm{cm}^{-3}]$",
            8,
            13,
            "magma",
        ),
        "bcon": (r"$log B$", None, None, "red"),
        "ucon": (r"$velo$", None, None, "green"),
        "log_Ehat_s": (
            r"$\log_{10}\,\widehat E\,[\mathrm{erg}\, \mathrm{cm}^{-3}]$",
            13.8,
            15.2,
            "inferno",
        ),
    }

    # First loop: interpolate all scalar and vector fields for each slice
    scalar_results = {}
    vector_results = {}

    for slice_name in slice_names:
        # Extract offset from slice name (e.g., "XY10" -> offset=10.0)
        offset = float(slice_name[2:]) if len(slice_name) > 2 else 0.0

        # Set up axis vectors for this slice
        slice_v = axis_vectors[slice_name[1]]  # First letter -> v
        slice_u = axis_vectors[slice_name[0]]  # Second letter -> u

        # Find the normal axis (the one not used in the slice)
        all_axes = {"X", "Y", "Z"}
        used_axes = {slice_name[0], slice_name[1]}
        normal_axis = list(all_axes - used_axes)[0]
        offset_vector = np.array(axis_vectors[normal_axis]) * offset

        X, Y, Z, duv = make_xyz(
            u=slice_u,
            v=slice_v,
            origin=offset_vector,
            ugrid=ugrid,
            vgrid=vgrid,
        )

        # Interpolate all scalars for this slice
        for scalar_name in scalar_names:
            density = getattr(sim, f"{scalar_name}_grid")
            scalars, _ = interpolate_slice_spherical(
                r,
                theta,
                phi,
                {scalar_name: np.where(r_mask, density, np.nan)},
                {},
                X,
                Y,
                Z,
                duv,
                method="linear",
            )
            scalar_results[(slice_name, scalar_name)] = scalars[scalar_name]

        # Interpolate all vectors for this slice
        for vector_name in vector_names:
            vector_field = getattr(sim, f"{vector_name}_grid")
            br = vector_field[:, :, :, 1]
            bth = vector_field[:, :, :, 2]
            bph = vector_field[:, :, :, 3]
            _, vectors = interpolate_slice_spherical(
                r,
                theta,
                phi,
                {},
                {vector_name: [np.where(r_mask, x, np.nan) for x in (br, bth, bph)]},
                X,
                Y,
                Z,
                duv,
                method="linear",
            )
            vector_results[(slice_name, vector_name)] = vectors[vector_name]

    print(f"interpolation complete: {args.d}. now drawing...")

    # Second loop: generate all figures from interpolated results
    for slice_name in slice_names:
        xlabel = r"$" + slice_name[0].lower() + r"\,[GM/c^2]$"
        ylabel = r"$" + slice_name[1].lower() + r"\,[GM/c^2]$"

        for scalar_name in scalar_names:
            for vector_name in vector_names:
                density_slice = scalar_results[(slice_name, scalar_name)]
                vecu_slice, vecv_slice = vector_results[(slice_name, vector_name)]

                # Get plot config for this scalar
                label, vmin, vmax, cmap = plot_config.get(
                    scalar_name, (scalar_name, None, None, "viridis")
                )
                # Get vector color from config
                vec_color = plot_config.get(
                    vector_name, (vector_name, None, None, "cyan")
                )[3]

                # Plot density + velocity quiver
                fig, ax = plt.subplots(figsize=(8, 7))

                # Apply log10 transformation if label contains "log"
                plot_data = density_slice
                # if "log" in label.lower():
                #     plot_data = np.log10(density_slice)

                pc = ax.pcolormesh(
                    ugrid,
                    vgrid,
                    plot_data,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                    cmap=cmap,
                    rasterized=True,
                )
                fig.colorbar(pc, ax=ax, label=label)

                ax.streamplot(
                    ugrid,
                    vgrid,
                    vecu_slice,
                    vecv_slice,
                    color=vec_color,
                    linewidth=0.6,
                    density=1.6,
                    arrowsize=0.8,
                )

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_xlim(-args.span, args.span)
                ax.set_ylim(-args.span, args.span)
                plt.tight_layout()

                outdir = args.outdir
                os.makedirs(outdir, exist_ok=True)

                base = os.path.splitext(os.path.basename(args.d))[0]
                prefix = f"{slice_name}slice_{scalar_name}_{vector_name}_"
                outfile = os.path.join(outdir, f"{prefix}{base}_fast.png")

                plt.savefig(outfile, bbox_inches="tight", dpi=400)
                plt.close()
                print(f"Saved: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D slice visualization")
    parser.add_argument("d", help="Input data file")
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Grid resolution (default: 1024)"
    )
    parser.add_argument(
        "--span", type=float, default=100, help="Grid span (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="outdir",
        type=str,
        default="slices",
        help="output dirr name. will be created if not exist",
    )
    parser.add_argument(
        "--scalar",
        nargs="+",
        default=["rho"],
        help="Scalar field name(s) (default: rho)",
    )
    parser.add_argument(
        "--vector",
        nargs="+",
        default=["bcon"],
        help="Vector field name(s) (default: bcon)",
    )
    parser.add_argument(
        "--slice",
        nargs="+",
        default=["XZ"],
        help="Slice plane(s): two letters from X,Y,Z optionally followed by offset (default: XZ)",
    )
    main(parser.parse_args())
