from typing import Literal, Sequence
import numpy as np
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count
import os
from scipy.interpolate import RegularGridInterpolator


def rthphi_xyz(r, th, phi):
    """Convert spherical (r, theta, phi) to Cartesian (x, y, z)."""
    x = r * np.sin(th) * np.cos(phi)
    y = r * np.sin(th) * np.sin(phi)
    z = r * np.cos(th)
    return x, y, z


def xyz_rthphi(x, y, z):
    """Convert Cartesian (x, y, z) to spherical (r, theta, phi)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return r, th, phi


def drthph_dxyz_3d_schw(r, th, phi, dr, dth, dphi, M=1.0):
    """
    Transform differentials from Schwarzschild spherical (r, θ, φ)
    to Cartesian (x, y, z).

    Schwarzschild metric accounts for spacetime curvature via
    the factor 1/sqrt(1 - 2M/r) for radial components.

    Parameters:
        r, th, phi: Schwarzschild coordinates
        dr, dth, dphi: Differentials in Schwarzschild coordinates
        M: Mass parameter (default 1)

    Returns:
        dx, dy, dz: Differentials in Cartesian coordinates
    """

    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Metric factor for radial coordinate
    g_rr_inv = 1.0 / np.sqrt(1.0 - 2.0 * M / r)

    # Jacobian components
    dx_dr = sin_th * cos_phi
    dx_dth = r * cos_th * cos_phi
    dx_dphi = -r * sin_th * sin_phi

    dy_dr = sin_th * sin_phi
    dy_dth = r * cos_th * sin_phi
    dy_dphi = r * sin_th * cos_phi

    dz_dr = cos_th
    dz_dth = -r * sin_th
    dz_dphi = 0.0

    # Apply Schwarzschild metric factor to dr terms
    dx = dx_dr * dr * g_rr_inv + dx_dth * dth + dx_dphi * dphi
    dy = dy_dr * dr * g_rr_inv + dy_dth * dth + dy_dphi * dphi
    dz = dz_dr * dr * g_rr_inv + dz_dth * dth + dz_dphi * dphi

    return dx, dy, dz


def drthph_dxyz_3d(r, th, phi, dr, dth, dphi):
    """
    Transform differentials from spherical (r, θ, φ)
    to Cartesian (x, y, z) without relativistic corrections.

    Parameters:
        r, th, phi: Spherical coordinates
        dr, dth, dphi: Differentials in spherical coordinates

    Returns:
        dx, dy, dz: Differentials in Cartesian coordinates
    """

    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Jacobian components
    dx_dr = sin_th * cos_phi
    dx_dth = r * cos_th * cos_phi
    dx_dphi = -r * sin_th * sin_phi

    dy_dr = sin_th * sin_phi
    dy_dth = r * cos_th * sin_phi
    dy_dphi = r * sin_th * cos_phi

    dz_dr = cos_th
    dz_dth = -r * sin_th
    dz_dphi = 0.0

    # Standard spherical Jacobian transformation
    dx = dx_dr * dr + dx_dth * dth + dx_dphi * dphi
    dy = dy_dr * dr + dy_dth * dth + dy_dphi * dphi
    dz = dz_dr * dr + dz_dth * dth + dz_dphi * dphi

    return dx, dy, dz


def make_xyz(
    u: np.ndarray,
    v: np.ndarray,
    ugrid: np.ndarray,
    vgrid: np.ndarray,
    origin: np.ndarray,
):
    """
    Example
    ```
    X, Y, Z, duv = make_xyz(
        u=[1, 1, 0],
        v=[-1, 1, 0],
        ugrid=np.linspace(-10, 10, 64),
        vgrid=np.linspace(-10, 10, 64),
        origin=[0, 0, 0],
    )
    ```
    """
    u = np.array(u) / np.linalg.norm(np.array(u))
    assert tuple(np.array(u).shape) == (3,)
    assert tuple(np.array(v).shape) == (3,)
    assert tuple(np.array(origin).shape) == (3,)
    v = np.array(v) / np.linalg.norm(np.array(v))
    if np.abs(np.dot(u, v)) > 1e-5:
        raise ValueError("U and V must be orhtogonal")

    origin = np.array(origin)

    # coordinates in figure space
    u_off, v_off = np.meshgrid(ugrid, vgrid)
    n = (ugrid.size, vgrid.size)
    # tensor with coordinates:
    # [ u pixel, v pixel, XYZ coordinate ]
    # for example, X = space_xyz[:,:,0]
    space_xyz = (
        origin.reshape([1, 1, 3])
        + u.reshape([1, 1, 3]) * u_off.reshape([n[0], n[1], 1])
        + v.reshape([1, 1, 3]) * v_off.reshape([n[0], n[1], 1])
    )

    return (
        space_xyz[:, :, 0],
        space_xyz[:, :, 1],
        space_xyz[:, :, 2],
        np.stack([u, v], axis=0),
    )


def interpolate_slice(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    scalars: dict[str, np.ndarray],
    vectors: dict[str, Sequence[np.ndarray]],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    duvdxyz: np.ndarray,
    method="linear",
):
    """
    Interpolate scalar and vector fields onto a plane y = yslice (x-z plane).

    ```
    X, Y, Z, duv = make_xyz(...)
    out_scalars, out_vectors = interpolate_slice(
        x, y, z, {"density": density}, {"myvec": [Vx, Vy, Vz]}, X, Y, Z, duv
    )
    u_comp, v_comp = out_vectors["myvec"]
    print(u_comp, v_comp )
    print(out_scalars["density"])
    ```
    """
    # Spherical to Cartesian coordinates for sample points
    points = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    out_scalars = {}
    for name, arr in scalars.items():
        vals = arr.flatten().copy()
        out_scalars[name] = griddata(points, vals, (X, Y, Z), method=method)

    out_vectors = {}
    for name, comps in vectors.items():
        veccomp_interp = np.stack(
            [
                griddata(points, comp.flatten(), (X, Y, Z), method=method)
                for comp in comps
            ],
            axis=0,
        )
        # ^ has dimension [3, nu, nv]
        vec_proj = [
            np.sum(veccomp_interp * proj.reshape([3, 1, 1]), axis=0) for proj in duvdxyz
        ]
        # ^ is Bu(u,v) and Bv(u,v)
        out_vectors[name] = vec_proj

    return out_scalars, out_vectors


def interpolate_slice_spherical(
    r: np.ndarray,
    theta: np.ndarray,
    fi: np.ndarray,
    scalars: dict[str, np.ndarray],
    vectors: dict[str, Sequence[np.ndarray]],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    duvdxyz: np.ndarray,
    gr: bool = True,
    method: Literal["linear", "nearest", "cubic"] = "linear",
):
    """
    Interpolate scalar and vector fields onto a plane y = yslice (x-z plane).

    ```
    X, Y, Z, duv = make_xyz(...)
    out_scalars, out_vectors = interpolate_slice(
        r, th, fi, {"density": density}, {"myvec": [Vr, Vth, Vfi]}, X, Y, Z, duv
    )
    u_comp, v_comp = out_vectors["myvec"]
    print(u_comp, v_comp )
    print(out_scalars["density"])
    ```
    """
    rslice, thslice, fislice = xyz_rthphi(X, Y, Z)
    interp_in = np.stack(
        [rslice.flatten(), thslice.flatten(), fislice.flatten()]
    ).transpose()

    vector_projection = drthph_dxyz_3d_schw if gr else drthph_dxyz_3d

    def interp_helper(arr):
        interpolator = RegularGridInterpolator(
            points=(r[:, 0, 0], theta[0, :, 0], fi[0, 0, :]),
            values=arr,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        return interpolator(interp_in).reshape(X.shape)

    out_scalars = {}
    for name, arr in scalars.items():
        out_scalars[name] = interp_helper(arr)

    out_vectors = {}
    for name, comps in vectors.items():
        comps = vector_projection(r, theta, fi, *comps)
        veccomp_interp = np.stack(
            [interp_helper(comp) for comp in comps],
            axis=0,
        )
        # ^ has dimension [3, nu, nv]
        vec_proj = [
            np.sum(veccomp_interp * proj.reshape([3, 1, 1]), axis=0) for proj in duvdxyz
        ]
        # ^ is Bu(u,v) and Bv(u,v)
        out_vectors[name] = vec_proj

    return out_scalars, out_vectors


if __name__ == "__main__":
    # #multiprocessing
    # n_proc = 4
    # with Pool(processes = n_proc) as p:
    #     results = p.map(interpolate_slice_spherical, range(1000))
    # Test the script
    ugrid = np.linspace(-2, 2, 33)  # np.array([-2, -1, 0, 1, 2])
    vgrid = np.linspace(-2, 2, 33)  # np.array([-2, -1, 0, 1, 2])

    X, Y, Z, t = make_xyz(
        u=[1, 1, 0],
        v=[-1, 1, 0],
        ugrid=ugrid,
        vgrid=vgrid,
        origin=[0, 0, 0],
    )

    print("X")
    print(X)

    print("Y")
    print(Y)

    print("Z")
    print(Z)

    print("t")
    print(t)

    x, y, z = np.meshgrid(
        np.linspace(-4, 4, 17), np.linspace(-4, 4, 17), np.linspace(-4, 4, 17)
    )
    density = np.abs(x) + np.abs(y) + np.abs(z)
    myvec = [x * 0 + 1, y, x * 0]
    out_scalars, out_vectors = interpolate_slice(
        x, y, z, {"density": density}, {"myvec": myvec}, X, Y, Z, t
    )
    u_comp, v_comp = out_vectors["myvec"]

    print("out_scalars")
    print(out_scalars["density"])

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(ugrid, vgrid, out_scalars["density"], cmap="viridis", shading="auto")
    plt.colorbar(label="density")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Density Heatmap")
    plt.savefig("density_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("out_vectors")
    for comp in out_vectors["myvec"]:
        print(comp)

    plt.figure(figsize=(8, 6))
    plt.streamplot(
        ugrid, vgrid, u_comp, v_comp, cmap="cool", density=1.5, arrowsize=1.5
    )
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title("Vector Field Streamlines")
    plt.savefig("streamlines.png", dpi=150, bbox_inches="tight")
    plt.close()


# interpolate_slice(*rthphi_xyz(r, theta, fi), {'rho': rho}, {'B': drthph_dxyz_3d_schw(r, theta, fi, Br, Bth, Bfi)}, *make_xyz(....))


# def interpolate(r, theta, phi, density, zslice):
#     # Convert spherical to Cartesian
#     x = r * np.sin(theta) * np.cos(phi)
#     y = r * np.sin(theta) * np.sin(phi)
#     z = r * np.cos(theta)

#     # Flatten arrays for griddata
#     points = np.array([x.flatten(), y.flatten(), z.flatten()]).T
#     values = density.flatten()

#     # Define 2D slice grid at z = zslice
#     xx = np.linspace(-100, 100, 100)  # z-axis range
#     yy = np.linspace(-100, 100, 100)  # y-axis range
#     Y, Z = np.meshgrid(yy, xx)
#     X = np.full_like(Y, zslice)

#     # Set values to NaN where r < 5
#     r_flat = r.flatten()
#     values[r_flat < 5] = np.nan

#     # Interpolate density on this slice
#     density_slice = griddata(points, values, (X, Y, Z), method="linear")

#     return Y, X, density_slice
