import numpy as np
import sympy as sp

# ================== PART A: Python Function for 3D ==================

def drthphi2dxyz(r, th, phi, dr, dth, dphi, M=1.0):
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
    g_rr_inv = 1.0 / np.sqrt(1.0 - 2.0*M / r)
    
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


# ================== PART B: SymPy Derivation ==================

def derive_schwarzschild_transform():
    """
    Derive the 3D Schwarzschild coordinate transformation symbolically.
    """
    
    # Define symbols
    r, th, phi, dr, dth, dphi, M = sp.symbols('r theta phi dr d_theta d_phi M', real=True, positive=True)
    
    print("=" * 70)
    print("SCHWARZSCHILD 3D COORDINATE TRANSFORMATION DERIVATION")
    print("=" * 70)
    
    # Coordinate transformations: spherical to Cartesian
    x = r * sp.sin(th) * sp.cos(phi)
    y = r * sp.sin(th) * sp.sin(phi)
    z = r * sp.cos(th)
    
    print("\n1. COORDINATE TRANSFORMATIONS:")
    print(f"   x = {x}")
    print(f"   y = {y}")
    print(f"   z = {z}")
    
    # Compute Jacobian matrix (partial derivatives)
    print("\n2. JACOBIAN MATRIX (∂x^i/∂y^a):")
    jacobian = sp.Matrix([
        [sp.diff(x, r), sp.diff(x, th), sp.diff(x, phi)],
        [sp.diff(y, r), sp.diff(y, th), sp.diff(y, phi)],
        [sp.diff(z, r), sp.diff(z, th), sp.diff(z, phi)]
    ])
    
    print("\nJ =")
    sp.pprint(jacobian)
    
    # Schwarzschild metric factor
    print("\n3. SCHWARZSCHILD METRIC FACTOR:")
    g_rr = 1 - 2*M/r
    g_rr_inv = 1/sp.sqrt(g_rr)
    print(f"   g_rr = {g_rr}")
    print(f"   1/√g_rr = {g_rr_inv}")
    
    # Modified dr with metric factor
    print("\n4. METRIC-ADJUSTED DIFFERENTIALS:")
    dr_adj = dr * g_rr_inv
    print(f"   dr_adjusted = dr / √(1 - 2M/r) = {dr_adj}")
    
    # Compute differentials: d(x,y,z) = J @ (dr_adj, dth, dphi)
    print("\n5. COORDINATE DIFFERENTIALS:")
    
    dx_sym = jacobian[0, 0] * dr_adj + jacobian[0, 1] * dth + jacobian[0, 2] * dphi
    dy_sym = jacobian[1, 0] * dr_adj + jacobian[1, 1] * dth + jacobian[1, 2] * dphi
    dz_sym = jacobian[2, 0] * dr_adj + jacobian[2, 1] * dth + jacobian[2, 2] * dphi
    
    print("\n   dx =")
    sp.pprint(sp.expand(dx_sym))
    
    print("\n   dy =")
    sp.pprint(sp.expand(dy_sym))
    
    print("\n   dz =")
    sp.pprint(sp.expand(dz_sym))
    
    # Simplified forms
    print("\n6. SIMPLIFIED FORMS (M=1):")
    dx_M1 = dx_sym.subs(M, 1)
    dy_M1 = dy_sym.subs(M, 1)
    dz_M1 = dz_sym.subs(M, 1)
    
    print("\n   dx =")
    sp.pprint(dx_M1)
    print("\n   dy =")
    sp.pprint(dy_M1)
    print("\n   dz =")
    sp.pprint(dz_M1)
    
    # For reference: 2D case (φ independent)
    print("\n7. 2D CASE (φ-independent, reduces to your original):")
    dx_2d = dx_sym.subs([(dphi, 0), (phi, 0), (M, 1)])
    dz_2d = dz_sym.subs([(dphi, 0), (phi, 0), (M, 1)])
    
    print("\n   dx = ")
    sp.pprint(sp.simplify(dx_2d))
    print("\n   dz = ")
    sp.pprint(sp.simplify(dz_2d))
    
    return dx_sym, dy_sym, dz_sym


# Run the derivation
if __name__ == "__main__":
    
    # Part B: Symbolic derivation
    print("\n" + "="*70)
    print("PART B: SYMBOLIC DERIVATION WITH SYMPY")
    print("="*70)
    dx, dy, dz = derive_schwarzschild_transform()
    
    # Part A: Numerical example
    print("\n\n" + "="*70)
    print("PART A: NUMERICAL EXAMPLE")
    print("="*70)
    
    r_val = 10.0
    th_val = np.pi / 4
    phi_val = np.pi / 6
    dr_val = 0.1
    dth_val = 0.05
    dphi_val = 0.02
    
    dx_num, dy_num, dz_num = drthphi2dxyz(r_val, th_val, phi_val, 
                                          dr_val, dth_val, dphi_val, M=1.0)
    
    print(f"\nInput (Schwarzschild coordinates):")
    print(f"   r = {r_val}, θ = {th_val:.4f}, φ = {phi_val:.4f}")
    print(f"   dr = {dr_val}, dθ = {dth_val}, dφ = {dphi_val}")
    
    print(f"\nOutput (Cartesian differentials):")
    print(f"   dx = {dx_num:.6f}")
    print(f"   dy = {dy_num:.6f}")
    print(f"   dz = {dz_num:.6f}")
