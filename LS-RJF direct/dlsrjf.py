
"""
lsrjf_direct.py
================
LS-RJF DIRECT -- computation of K_eff for fiber-reinforced composites.


Compares LS-RJF vs FFT (reference) vs FEniCS.
FFT is always the reference.

Series:
  1  -- Baseline validation + T field plots (z=0.5 slice)
  2  -- Volume fraction sweep
  3  -- Aspect ratio sweep
  4  -- Fiber orientation sweep
  5  -- FEM mesh resolution convergence
  6  -- Fiber conductivity sweep
  7  -- Reference medium k0 sweep
  8  -- Hyperparameter sensitivity (Ng, Nc, n_layers)
"""
import numpy as np
import torch
import time
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares
from torch.func import jacfwd

sys.path.insert(0, '.')
from rve import RVE_Cylinder

# ── FEniCS ────────────────────────────────────────────────────────────────────
try:
    from fenics import *
    from dolfin import *
    set_log_level(LogLevel.WARNING)
    HAS_FENICS = True
except ImportError:
    print("WARNING: FEniCS not found -- FEniCS series will be skipped")
    HAS_FENICS = False

torch.set_default_dtype(torch.float32)
torch.manual_seed(1234)
np.random.seed(1234)

# ── Global constants ──────────────────────────────────────────────────────────
K_CLAY  = 0.651   # matrix conductivity  [W/(m·K)]
K_FIBER = 0.29    # fiber conductivity   [W/(m·K)]
K0      = (K_CLAY + K_FIBER) / 2.   # default reference medium
LX = LY = LZ = 1.0
DIRS = ['x', 'y', 'z']

# Default hyperparameters (same as inverse)
ARCH  = (6, 20, 20,1)
N_COL = 500
N_GRID = 31

# ── Utilities ─────────────────────────────────────────────────────────────────
def sep(title="", w=72):
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + " " + title + " " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


def print_tensor(K, label="K_eff", K_ref=None):
    print(f"\n  {label}:")
    for d in range(3):
        row = "  ".join(f"{K[d, j]:9.5f}" for j in range(3))
        print(f"    [{row}]")
    if K_ref is not None:
        e = keff_errors(K, K_ref)
        print(f"  errors vs FFT: exx={e['e_xx']:.2f}%  "
              f"eyy={e['e_yy']:.2f}%  ezz={e['e_zz']:.2f}%  "
              f"emax={e['e_max']:.2f}%")


def keff_errors(K, K_ref):
    ediag = {f'e_{d}{d}': abs(K[i, i] - K_ref[i, i]) / (abs(K_ref[i, i]) + 1e-12) * 100
             for i, d in enumerate(DIRS)}
    eoff = {
        'e_xy': abs(K[0, 1] - K_ref[0, 1]) / (abs(K_ref[0, 0]) + 1e-12) * 100,
        'e_xz': abs(K[0, 2] - K_ref[0, 2]) / (abs(K_ref[0, 0]) + 1e-12) * 100,
        'e_yz': abs(K[1, 2] - K_ref[1, 2]) / (abs(K_ref[1, 1]) + 1e-12) * 100,
    }
    eall = {**ediag, **eoff}
    eall['e_max'] = max(eall.values())
    return eall


def print_row(cfg, method, K, K_ref=None, nfev=None, t=None):
    kxx, kyy, kzz = K[0, 0], K[1, 1], K[2, 2]
    kxy, kxz, kyz = K[0, 1], K[0, 2], K[1, 2]
    s = f"  {cfg:30s} {method:8s}  "
    s += f"Kxx={kxx:.5f} Kyy={kyy:.5f} Kzz={kzz:.5f}  "
    s += f"Kxy={kxy:.5f} Kxz={kxz:.5f} Kyz={kyz:.5f}"
    if K_ref is not None:
        e = keff_errors(K, K_ref)
        s += (f"  exx={e['e_xx']:.2f}% eyy={e['e_yy']:.2f}%"
              f" ezz={e['e_zz']:.2f}% emax={e['e_max']:.2f}%")
    if nfev is not None:
        s += f"  nfev={nfev}"
    if t is not None:
        s += f"  t={t:.1f}s"
    print(s)


def print_bounds(vf, kf=K_FIBER, km=K_CLAY):
    voigt = vf * kf + (1 - vf) * km
    reuss = 1.0 / (vf / kf + (1 - vf) / km)
    mg    = km * (1 + 3 * vf * (kf - km) / (2 * km + kf))
    print(f"  Vf={vf:.4f}  Voigt={voigt:.5f}  "
          f"Reuss={reuss:.5f}  Maxwell-Garnett={mg:.5f}")
    return voigt, reuss, mg

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (same as inverse)
# ══════════════════════════════════════════════════════════════════════════════
def make_grid(N):
    x = np.linspace(0, LX, N, dtype=np.float32)
    Xg, Yg, Zg = np.meshgrid(x, x, x, indexing='ij')
    return np.hstack([Xg.ravel()[:, None],
                      Yg.ravel()[:, None],
                      Zg.ravel()[:, None]])


def build_gamma0(N, k0):
    freq = np.fft.fftfreq(N, d=1. / N)
    fx, fy, fz = np.meshgrid(2 * np.pi * freq / LX,
                              2 * np.pi * freq / LX,
                              2 * np.pi * freq / LX, indexing='ij')
    xi2 = fx ** 2 + fy ** 2 + fz ** 2
    xi2_safe = xi2.copy(); xi2_safe[0, 0, 0] = 1.
    ker = torch.tensor(1. / (k0 * xi2_safe), dtype=torch.complex64)
    ker[0, 0, 0] = 0.
    XI = torch.tensor(np.stack([fx, fy, fz], axis=-1), dtype=torch.complex64)

    def gamma0(tau):
        t3 = tau.to(torch.complex64).reshape(N, N, N, 3)
        th = torch.fft.fftn(t3.permute(3, 0, 1, 2),
                             dim=(1, 2, 3)).permute(1, 2, 3, 0)
        xd = (XI * th).sum(-1)
        return torch.stack(
            [torch.fft.ifftn(XI[:, :, :, a] * ker * xd,
                             dim=(0, 1, 2)).real
             for a in range(3)], dim=-1).reshape(N ** 3, 3)
    return gamma0


def fft_keff(fmask_np, kf_val, km_val, N, k0):
    """
    Moulinec-Suquet FFT solver.
    Returns K_eff (3x3) and eps_list (3 arrays of shape (N^3,3)).
    """
    freq = np.fft.fftfreq(N, d=1. / N)
    fx, fy, fz = np.meshgrid(2 * np.pi * freq / LX,
                              2 * np.pi * freq / LX,
                              2 * np.pi * freq / LX, indexing='ij')
    xi2 = fx ** 2 + fy ** 2 + fz ** 2; xi2[0, 0, 0] = 1.
    ker = 1. / (k0 * xi2); ker[0, 0, 0] = 0.
    XI  = np.stack([fx, fy, fz], axis=-1)
    k_f = np.where(fmask_np, kf_val, km_val).reshape(N, N, N).astype(np.float64)
    K   = np.zeros((3, 3))
    eps_list = []
    for d in range(3):
        E   = np.zeros(3); E[d] = 1.
        eps = np.zeros((N, N, N, 3)); eps[:, :, :, :] = E
        for _ in range(500):
            tau  = (k_f[:, :, :, None] - k0) * eps
            th   = np.fft.fftn(tau, axes=(0, 1, 2))
            xd   = sum(XI[:, :, :, a] * th[:, :, :, a] for a in range(3))
            g0   = np.zeros_like(tau, dtype=complex)
            for a in range(3):
                g0[:, :, :, a] = np.fft.ifftn(XI[:, :, :, a] * ker * xd)
            en = E - np.real(g0)
            if np.max(np.abs(en - eps)) < 1e-10: break
            eps = en
        K[:, d] = np.mean((k_f[:, :, :, None] * eps).reshape(-1, 3), axis=0)
        eps_list.append(eps.reshape(N ** 3, 3).astype(np.float32))
    return K, eps_list


def unpack(p_flat, arch):
    ps = []; ip = 0
    for i in range(len(arch) - 1):
        ni, no = arch[i], arch[i + 1]
        ps.append(p_flat[ip:ip + ni * no].reshape(ni, no)); ip += ni * no
        ps.append(p_flat[ip:ip + no].reshape(1, no)); ip += no
    return ps


def net_eps(params, x, y, z, G):
    k = 2. * np.pi / LX
    feat = torch.cat([torch.sin(k * x), torch.cos(k * x),
                      torch.sin(k * y), torch.cos(k * y),
                      torch.sin(k * z), torch.cos(k * z)], dim=1)
    hx = torch.cat([k * torch.cos(k * x), -k * torch.sin(k * x),
                    0 * y, 0 * y, 0 * z, 0 * z], dim=1)
    hy = torch.cat([0 * x, 0 * x, k * torch.cos(k * y),
                    -k * torch.sin(k * y), 0 * z, 0 * z], dim=1)
    hz = torch.cat([0 * x, 0 * x, 0 * y, 0 * y,
                    k * torch.cos(k * z), -k * torch.sin(k * z)], dim=1)
    H = feat
    for i in range(0, len(params) - 2, 2):
        W = params[i]; b = params[i + 1]
        pre = H @ W + b; act = torch.tanh(pre); dp = 1. - act ** 2
        hx = dp * (hx @ W); hy = dp * (hy @ W); hz = dp * (hz @ W)
        H  = act
    u = H @ params[-2] + params[-1]
    du_dx = hx @ params[-2]
    du_dy = hy @ params[-2]
    du_dz = hz @ params[-2]
    return u, torch.cat([-(G[0] + du_dx),
                         -(G[1] + du_dy),
                         -(G[2] + du_dz)], dim=1)

# ══════════════════════════════════════════════════════════════════════════════
# LS-RJF DIRECT SOLVER
# ══════════════════════════════════════════════════════════════════════════════
class DirectLSRJF:
    """
    LS-RJF for direct thermal homogenization.
    Solves for theta* minimizing the Lippmann-Schwinger residual.
    Returns K_eff computed from the converged field.
    """
    def __init__(self, X_full, fmask_np, kf, km, k0,
                 arch=ARCH, N_col=N_COL):
        self.arch  = list(arch)
        self.k0    = k0
        self.kf    = kf; self.km = km
        self.gamma0 = build_gamma0(int(round(len(X_full) ** (1/3))), k0)
        self.N     = int(round(len(X_full) ** (1/3)))

        # precompute dk field
        dk_np = (np.where(fmask_np, kf, km) - k0).astype(np.float32)
        self.dk = torch.tensor(dk_np)
        self.fm = torch.tensor(fmask_np.astype(np.float32))
        self.mf = 1. - self.fm

        # full grid tensors
        self.x_f = torch.tensor(X_full[:, 0:1])
        self.y_f = torch.tensor(X_full[:, 1:2])
        self.z_f = torch.tensor(X_full[:, 2:3])

        # collocation subset
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_full), N_col, replace=False)
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.x_c = self.x_f[self.idx]
        self.y_c = self.y_f[self.idx]
        self.z_c = self.z_f[self.idx]
        self.dk_c = self.dk[self.idx]

        self.Gs = [torch.tensor([1., 0., 0.]),
                   torch.tensor([0., 1., 0.]),
                   torch.tensor([0., 0., 1.])]

        # init params for all 3 directions
        n_per = sum(arch[i] * arch[i+1] + arch[i+1]
                    for i in range(len(arch)-1))
        self.n_per = n_per
        params0 = []
        for d in range(3):
            torch.manual_seed(1234 + d)
            for i in range(len(arch) - 1):
                std = float(np.sqrt(2. / (arch[i] + arch[i+1])))
                params0.append(torch.randn(arch[i], arch[i+1]) * std)
                params0.append(torch.zeros(1, arch[i+1]))
        self.p0 = torch.cat([p.reshape(-1) for p in params0]).numpy()
        self._nfev = 0

    def _g0_dk_eps(self, p_t, d):
        """Precompute Gamma0 * dk * eps on full grid for direction d."""
        with torch.no_grad():
            G = self.Gs[d]
            params = unpack(p_t[d * self.n_per:(d+1) * self.n_per], self.arch)
            _, eps_full = net_eps(params, self.x_f, self.y_f, self.z_f, G)
            return self.gamma0(self.dk[:, None] * eps_full)[self.idx].detach()

    def R_np(self, p):
        p_t = torch.tensor(p, dtype=torch.float32)
        r_all = []
        with torch.no_grad():
            for d, G in enumerate(self.Gs):
                g0 = self._g0_dk_eps(p_t, d)
                params = unpack(p_t[d*self.n_per:(d+1)*self.n_per], self.arch)
                u_c, eps_c = net_eps(params, self.x_c, self.y_c, self.z_c, G)
                E = -G
                r_LS  = (eps_c + g0 - E).reshape(-1)
                r_u0  = u_c.mean(0).reshape(-1)
                r_all.append(torch.cat([r_LS, r_u0]))
        r = torch.cat(r_all)
        loss = (r**2).mean().item()
        print(f"  R nfev={self._nfev:4d}  loss={loss:.3e}", flush=True)
        self._nfev += 1
        return r.numpy()

    def J_np(self, p):
        p_t = torch.tensor(p, dtype=torch.float32)
        Nc  = self.x_c.shape[0]
        n_per = self.n_per
        # residual size per direction: Nc*3 + 1
        n_r_d   = Nc * 3 + 1
        n_r_tot = 3 * n_r_d
        J_full = np.zeros((n_r_tot, len(p)), dtype=np.float32)

        for d, G in enumerate(self.Gs):
            g0 = self._g0_dk_eps(p_t, d)
            p_one = p_t[d * n_per:(d+1) * n_per]

            def res_d(p_one, g0=g0, G=G,
                      x_c=self.x_c, y_c=self.y_c, z_c=self.z_c,
                      arch=self.arch):
                E = -G
                params = unpack(p_one, arch)
                u_c, eps_c = net_eps(params, x_c, y_c, z_c, G)
                r_LS = (eps_c + g0 - E).reshape(-1)
                r_u0 = u_c.mean(0).reshape(-1)
                return torch.cat([r_LS, r_u0])

            Jd = jacfwd(res_d)(p_one)
            r0 = d * n_r_d;  r1 = r0 + n_r_d
            c0 = d * n_per;  c1 = c0 + n_per
            J_full[r0:r1, c0:c1] = Jd.detach().numpy()
        return J_full

    def solve(self):
        """Solve and return (K_eff, eps_list, nfev, t)."""
        print("  [warmup...]", end=' ', flush=True)
        _ = self.J_np(self.p0.copy())
        print("done")
        t0 = time.perf_counter()
        res = least_squares(fun=self.R_np, jac=self.J_np,
                            x0=self.p0, method='trf')
        t = time.perf_counter() - t0

        # compute K_eff from converged field on full grid
        p_t = torch.tensor(res.x, dtype=torch.float32)
        K = np.zeros((3, 3))
        eps_list = []
        with torch.no_grad():
            k_full = (self.fm * self.kf + self.mf * self.km)
            for d, G in enumerate(self.Gs):
                params = unpack(p_t[d*self.n_per:(d+1)*self.n_per], self.arch)
                _, eps_full = net_eps(params, self.x_f, self.y_f, self.z_f, G)
                K[:, d] = -(k_full[:, None] * eps_full).mean(0).numpy()
                eps_list.append(eps_full.numpy())
        return K, eps_list, res.nfev, t


def run_lsrjf_direct(X_full, fmask_np, kf, km, k0=None,
                     arch=ARCH, N_col=N_COL):
    if k0 is None:
        k0 = (kf + km) / 2.
    solver = DirectLSRJF(X_full, fmask_np, kf, km, k0,
                         arch=arch, N_col=N_col)
    return solver.solve()

# ══════════════════════════════════════════════════════════════════════════════
# FEniCS DIRECT SOLVER
# ══════════════════════════════════════════════════════════════════════════════
if HAS_FENICS:
    class PeriodicBC(SubDomain):
        def __init__(self, L):
            self.L = L; super().__init__()
        def inside(self, x, on_boundary):
            return on_boundary and (
                (near(x[0], 0.) and not near(x[0], self.L)) or
                (near(x[1], 0.) and not near(x[1], self.L)) or
                (near(x[2], 0.) and not near(x[2], self.L)))
        def map(self, x, y):
            y[0] = x[0] - self.L if near(x[0], self.L) else x[0]
            y[1] = x[1] - self.L if near(x[1], self.L) else x[1]
            y[2] = x[2] - self.L if near(x[2], self.L) else x[2]

    def run_fenics_direct(rve, resolution=20, kf=K_FIBER, km=K_CLAY):
        """FEniCS direct homogenization. Returns K_eff (3x3) and t (s)."""
        mesh = BoxMesh(Point(0,0,0), Point(LX,LX,LX),
                       resolution, resolution, resolution)
        periodic_bc = PeriodicBC(LX)
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        R_e = FiniteElement("R",  mesh.ufl_cell(), 0)
        W  = FunctionSpace(mesh, MixedElement([P1, R_e]),
                           constrained_domain=periodic_bc)

        # Build DG0 conductivity field
        V0 = FunctionSpace(mesh, "DG", 0)
        k_func = Function(V0)
        dm = V0.dofmap(); nc = mesh.num_cells()
        centers = np.array([Cell(mesh, i).midpoint().array()
                             for i in range(nc)], dtype=np.float32)
        fmask = (rve.get_points_in_fiber_mask(centers)
                 if hasattr(rve, 'get_points_in_fiber_mask')
                 else rve.get_fiber_mask(centers))
        k_vals = np.where(fmask, kf, km).astype(float)
        k_vec  = np.zeros(V0.dim())
        for i in range(nc):
            k_vec[dm.cell_dofs(i)[0]] = k_vals[i]
        k_func.vector().set_local(k_vec); k_func.vector().apply("insert")

        vol = assemble(Constant(1.) * dx(domain=mesh))
        K   = np.zeros((3, 3))
        eps_list = []
        t0 = time.perf_counter()
        for d, E_vec in enumerate([[1,0,0],[0,1,0],[0,0,1]]):
            E = Constant(E_vec)
            (u, lam) = TrialFunctions(W); (v, mu) = TestFunctions(W)
            a = (inner(k_func * grad(u), grad(v)) * dx
                 + lam * v * dx + mu * u * dx)
            Lf = -inner(k_func * E, grad(v)) * dx
            w_h = Function(W)
            solve(a == Lf, w_h,
                  solver_parameters={"linear_solver": "mumps"})
            u_h, _ = w_h.split(deepcopy=True)
            V3 = VectorFunctionSpace(mesh, "DG", 0)
            q  = project(-k_func * (E + grad(u_h)), V3)
            K[:, d] = -np.array([assemble(q[i] * dx) / vol
                                  for i in range(3)])
            eps_proj = project(E + grad(u_h), V3)
            eps_list.append(eps_proj)
        t = time.perf_counter() - t0
        return K, eps_list, t

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 1 PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_T_fields_series1(X_full, N_GRID,
                          eps_fft, eps_lsrjf,
                          kf=K_FIBER, km=K_CLAY, fmask_np=None,
                          eps_fen=None, centers_fen=None):
    """
    Series 1 -- one PNG per loading direction.
    Panels: microstructure | FFT u | LS-RJF u | FEniCS u (if available).
    All u panels share the same colorbar. z = N//2 slice.
    """
    from scipy.interpolate import griddata
    N   = N_GRID
    x1d = np.linspace(0, LX, N)
    Xg, Yg = np.meshgrid(x1d, x1d, indexing='ij')
    iz = N // 2
    z_target = iz / max(N - 1, 1) * LZ

    k_2d = (np.where(fmask_np, kf, km).reshape(N, N, N)[:, :, iz]
            if fmask_np is not None else None)

    freq = np.fft.fftfreq(N, d=1. / N)
    fx, fy, fz = np.meshgrid(2*np.pi*freq/LX, 2*np.pi*freq/LX,
                              2*np.pi*freq/LX, indexing='ij')
    xi2 = fx**2 + fy**2 + fz**2; xi2[0, 0, 0] = 1.
    XI  = np.stack([fx, fy, fz], axis=-1)

    def poisson(grad_u_flat):
        gu_hat  = np.fft.fftn(grad_u_flat.reshape(N, N, N, 3), axes=(0,1,2))
        div_hat = np.sum(1j * XI * gu_hat, axis=-1)
        u_hat   = div_hat / (-xi2 + 0j); u_hat[0, 0, 0] = 0.
        return np.real(np.fft.ifftn(u_hat))[:, :, iz]

    def u_fft_d(eps, d):
        E = np.zeros(3); E[d] = 1.
        return poisson(eps.astype(float) - E[None, :])

    def u_lsrjf_d(eps, d):
        E = np.zeros(3); E[d] = 1.
        return poisson(-eps.astype(float) - E[None, :])

    def u_fen_d(eps_func, d, ref):
        if eps_func is None or centers_fen is None:
            return None
        try:
            E = np.zeros(3); E[d] = 1.
            eps_vals = np.array([[float(eps_func(c)[j])
                                   for j in range(3)]
                                  for c in centers_fen])
            gu = eps_vals - E[None, :]
            dz   = np.abs(centers_fen[:, 2] - z_target)
            tol  = np.percentile(dz, max(3., 200./len(centers_fen)*100))
            msk  = dz <= tol
            if msk.sum() < 9:
                return None
            pts = np.column_stack([centers_fen[msk, 0],
                                   centers_fen[msk, 1]])
            tgt = np.column_stack([Xg.ravel(), Yg.ravel()])
            u2d = griddata(pts, gu[msk, d], tgt,
                           method='linear', fill_value=np.nan).reshape(N, N)
            ok = ~np.isnan(u2d)
            if int(ok.sum()) > 4 and ref is not None:
                s = np.nanstd(ref) / (np.nanstd(u2d[ok]) + 1e-12)
                u2d = (u2d - np.nanmean(u2d)) * s + np.nanmean(ref)
            return u2d
        except Exception as e:
            print(f"  FEniCS u recovery error: {e}")
            return None

    has_fen = (HAS_FENICS and eps_fen is not None
               and centers_fen is not None)

    for d, dlabel in enumerate(['x', 'y', 'z']):
        uf  = u_fft_d(eps_fft[d], d)
        ulr = u_lsrjf_d(eps_lsrjf[d], d)
        ufe = u_fen_d(eps_fen[d] if has_fen else None, d, uf)               if has_fen else None

        vmin = min(uf.min(), ulr.min())
        vmax = max(uf.max(), ulr.max())

        panels = []
        if k_2d is not None:
            panels.append(('Microstructure k(x)', k_2d,
                            'RdBu_r', kf, km, 'k [W/mK]'))
        panels.append(('FFT  u=T-E.x', uf,
                        'coolwarm', vmin, vmax, 'u [K]'))
        panels.append(('LS-RJF  u=T-E.x', ulr,
                        'coolwarm', vmin, vmax, 'u [K]'))
        if ufe is not None:
            panels.append(('FEniCS  u=T-E.x', ufe,
                            'coolwarm', vmin, vmax, 'u [K]'))

        n   = len(panels)
        fig = plt.figure(figsize=(4.8 * n, 5.2))
        gs  = GridSpec(1, n, wspace=0.40,
                       left=0.06, right=0.97, top=0.87, bottom=0.10)
        fig.suptitle(
            f'Series 1 -- u=T-E.x  z=0.5  loading e_{dlabel}',
            fontsize=12, fontweight='bold')

        for col, (title, field, cmap, vlo, vhi, clab) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            im = ax.pcolormesh(Xg, Yg, field, cmap=cmap,
                               vmin=vlo, vmax=vhi, shading='nearest')
            plt.colorbar(im, ax=ax, shrink=0.85).set_label(clab, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y' if col == 0 else '')
            ax.set_aspect('equal')

        fname = f'./direction_{dlabel}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {fname}')

def series1_baseline(N_grid=N_GRID, N_col=N_COL, FEM_RES=30,
                     arch=ARCH, kf=K_FIBER, km=K_CLAY):
    sep("SERIES 1 -- Baseline Validation")
    k0 = (kf + km) / 2.
    print(f"  Config: Vf=10%, alpha=5, O-y  kf={kf} km={km}  "
          f"k0={k0:.4f}  Ng={N_grid}  Nc={N_col}  arch={arch}")

    rve = RVE_Cylinder(
        n_inclusions=20, volume_fraction=0.10,
        RVE_size=[LX,LX,LX], aspect_ratio=5.0,
        orientation_type='axis', orientation_params={'axis': 'y'},
        seed=1234)
    rve.generate_RSA(max_attempts=5000)
    vf = rve.actual_volume_fraction
    voigt, reuss, mg = print_bounds(vf, kf, km)

    X_full  = make_grid(N_grid)
    fmask_np = rve.get_fiber_mask(X_full)

    # ── FFT reference ──────────────────────────────────────────────────────
    print("\n  [FFT reference]")
    t0 = time.perf_counter()
    K_fft, eps_fft = fft_keff(fmask_np, kf, km, N_grid, k0)
    t_fft = time.perf_counter() - t0
    print_tensor(K_fft, "K_eff FFT")
    print_row(f"Vf={vf:.3f} alpha=5 O-y", "FFT", K_fft, t=t_fft)

    # ── LS-RJF ─────────────────────────────────────────────────────────────
    print("\n  [LS-RJF direct]")
    K_lr, eps_lr, nfev_lr, t_lr = run_lsrjf_direct(
        X_full, fmask_np, kf, km, k0, arch=arch, N_col=N_col)
    print_tensor(K_lr, "K_eff LS-RJF", K_ref=K_fft)
    print_row(f"Vf={vf:.3f} alpha=5 O-y", "LS-RJF",
              K_lr, K_fft, nfev_lr, t_lr)

    # ── FEniCS ─────────────────────────────────────────────────────────────
    K_fen = None; eps_fen = None; centers_fen = None
    if HAS_FENICS:
        print("\n  [FEniCS]")
        K_fen, eps_fen, t_fen = run_fenics_direct(rve, resolution=FEM_RES,
                                                    kf=kf, km=km)
        print_tensor(K_fen, "K_eff FEniCS", K_ref=K_fft)
        print_row(f"Vf={vf:.3f} alpha=5 O-y", "FEniCS",
                  K_fen, K_fft, t=t_fen)
        # get cell centers for FEniCS interpolation in plot
        try:
            from fenics import BoxMesh, Point, Cell
            mesh_tmp = BoxMesh(Point(0,0,0), Point(LX,LX,LX),
                               FEM_RES, FEM_RES, FEM_RES)
            centers_fen = np.array([Cell(mesh_tmp, i).midpoint().array()
                                     for i in range(mesh_tmp.num_cells())],
                                    dtype=np.float32)
        except Exception:
            centers_fen = None

    # ── T field plots ───────────────────────────────────────────────────────
    print("\n  Generating T field plots (z=0.5 slice)...")
    plot_T_fields_series1(X_full, N_grid, eps_fft, eps_lr,
                          kf=kf, km=km, fmask_np=fmask_np,
                          eps_fen=eps_fen, centers_fen=centers_fen)

    # ── Physics check ───────────────────────────────────────────────────────
    print(f"\n  Physics check (kf={kf} < km={km} => Kyy > Kxx for O-y):")
    for name, K in [("FFT", K_fft), ("LS-RJF", K_lr), ("FEniCS", K_fen)]:
        if K is None: continue
        ok = K[1, 1] > K[0, 0]
        print(f"    {name:8s}: Kxx={K[0,0]:.5f}  Kyy={K[1,1]:.5f}  "
              f"Kyy>Kxx {'OK' if ok else 'FAIL'}")

    return {'K_fft': K_fft, 'K_lr': K_lr, 'K_fen': K_fen,
            'eps_fft': eps_fft, 'eps_lr': eps_lr,
            'vf': vf, 'voigt': voigt, 'reuss': reuss}

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 2 -- VOLUME FRACTION SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def series2_vf_sweep(vf_list=None, N_grid=N_GRID, N_col=N_COL,
                     FEM_RES=30, arch=ARCH):
    if vf_list is None:
        vf_list = [0.05, 0.10, 0.15, 0.20, 0.25]
    sep("SERIES 2 -- Volume Fraction Sweep")
    print(f"  Vf list: {vf_list}  alpha=5, O-y  "
          f"Ng={N_grid}  Nc={N_col}  arch={arch}")

    for vf_t in vf_list:
        print(f"\n  == Vf={vf_t} ==")
        rve = RVE_Cylinder(
            n_inclusions=20, volume_fraction=vf_t,
            RVE_size=[LX,LX,LX], aspect_ratio=5.0,
            orientation_type='axis', orientation_params={'axis': 'y'},
            seed=1234)
        rve.generate_RSA(max_attempts=8000)
        vf = rve.actual_volume_fraction
        k0 = (K_FIBER + K_CLAY) / 2.
        voigt, reuss, mg = print_bounds(vf)

        X_full   = make_grid(N_grid)
        fmask_np = rve.get_fiber_mask(X_full)

        print("  [FFT]")
        K_fft, _, = fft_keff(fmask_np, K_FIBER, K_CLAY, N_grid, k0)
        print_row(f"Vf={vf:.3f}", "FFT", K_fft)

        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, K_FIBER, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(f"Vf={vf:.3f}", "LS-RJF", K_lr, K_fft, nfev, t)

        if HAS_FENICS:
            print("  [FEniCS]")
            K_fen, _, t_fen = run_fenics_direct(rve, resolution=FEM_RES)
            print_row(f"Vf={vf:.3f}", "FEniCS", K_fen, K_fft, t=t_fen)

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 3 -- ASPECT RATIO SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def series3_ar_sweep(ar_list=None, N_grid=N_GRID, N_col=N_COL,
                     FEM_RES=30, arch=ARCH):
    if ar_list is None:
        ar_list = [ 1,2,5,10,20.0]
    sep("SERIES 3 -- Aspect Ratio Sweep")
    print(f"  AR list: {ar_list}  Vf=10%, O-y  "
          f"Ng={N_grid}  Nc={N_col}  arch={arch}")

    for ar in ar_list:
        print(f"\n  == alpha={ar} ==")
        rve = RVE_Cylinder(
            n_inclusions=20, volume_fraction=0.10,
            RVE_size=[LX,LX,LX], aspect_ratio=ar,
            orientation_type='axis', orientation_params={'axis': 'y'},
            seed=1234)
        rve.generate_RSA(max_attempts=8000)
        vf = rve.actual_volume_fraction
        k0 = (K_FIBER + K_CLAY) / 2.
        print_bounds(vf)

        X_full   = make_grid(N_grid)
        fmask_np = rve.get_fiber_mask(X_full)

        print("  [FFT]")
        K_fft, _ = fft_keff(fmask_np, K_FIBER, K_CLAY, N_grid, k0)
        print_row(f"ar={ar}", "FFT", K_fft)
        print(f"  Anisotropy Kyy/Kxx = {K_fft[1,1]/K_fft[0,0]:.4f}")

        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, K_FIBER, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(f"ar={ar}", "LS-RJF", K_lr, K_fft, nfev, t)
        print(f"  Anisotropy Kyy/Kxx = {K_lr[1,1]/K_lr[0,0]:.4f}")

        if HAS_FENICS:
            print("  [FEniCS]")
            K_fen, _, t_fen = run_fenics_direct(rve, resolution=FEM_RES)
            print_row(f"ar={ar}", "FEniCS", K_fen, K_fft, t=t_fen)

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 4 -- FIBER ORIENTATION SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def series4_orientation(N_grid=N_GRID, N_col=N_COL,
                        FEM_RES=30, arch=ARCH):
    sep("SERIES 4 -- Fiber Orientation Sweep")
    print(f"  Vf=10%, alpha=5  Ng={N_grid}  Nc={N_col}  arch={arch}")

    configs = [
        ('O-x',    'axis',   {'axis': 'x'}),
        ('O-y',    'axis',   {'axis': 'y'}),
        ('O-z',    'axis',   {'axis': 'z'}),
        ('O-rand', 'random', {}),
    ]
    results = {}
    for case, otype, oparams in configs:
        print(f"\n  == {case} ==")
        rve = RVE_Cylinder(
            n_inclusions=20, volume_fraction=0.10,
            RVE_size=[LX,LX,LX], aspect_ratio=5.0,
            orientation_type=otype, orientation_params=oparams,
            seed=1234)
        rve.generate_RSA(max_attempts=8000)
        vf = rve.actual_volume_fraction
        k0 = (K_FIBER + K_CLAY) / 2.
        print_bounds(vf)

        X_full   = make_grid(N_grid)
        fmask_np = rve.get_fiber_mask(X_full)

        print("  [FFT]")
        K_fft, _ = fft_keff(fmask_np, K_FIBER, K_CLAY, N_grid, k0)
        print_row(case, "FFT", K_fft)

        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, K_FIBER, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(case, "LS-RJF", K_lr, K_fft, nfev, t)
        results[case] = {'K_fft': K_fft, 'K_lr': K_lr}

        if HAS_FENICS:
            print("  [FEniCS]")
            K_fen, _, t_fen = run_fenics_direct(rve, resolution=FEM_RES)
            print_row(case, "FEniCS", K_fen, K_fft, t=t_fen)

    # Permutation symmetry check
    print("\n  Permutation symmetry check (Kxx(O-x) == Kyy(O-y) == Kzz(O-z)):")
    for m in ('K_fft', 'K_lr'):
        kox = results.get('O-x',  {}).get(m)
        koy = results.get('O-y',  {}).get(m)
        koz = results.get('O-z',  {}).get(m)
        if kox is None or koy is None or koz is None: continue
        name = 'FFT' if m == 'K_fft' else 'LS-RJF'
        ok = (abs(kox[0,0] - koy[1,1]) < 0.005 and
              abs(kox[0,0] - koz[2,2]) < 0.005)
        print(f"  {name:8s}: Kxx(O-x)={kox[0,0]:.5f}  "
              f"Kyy(O-y)={koy[1,1]:.5f}  "
              f"Kzz(O-z)={koz[2,2]:.5f}  "
              f"{'OK' if ok else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 5 -- FEM MESH RESOLUTION CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════
def series5_resolution(resolutions=None, N_col=N_COL, arch=ARCH):
    if resolutions is None:
        resolutions = [10, 15, 20, 30]
    sep("SERIES 5 -- Resolution Convergence (FFT + LS-RJF + FEniCS)")
    print(f"  FEM resolutions: {resolutions}")
    print(f"  FFT/LS-RJF: N_grid = resolution + 1 for each case")

    rve = RVE_Cylinder(
        n_inclusions=20, volume_fraction=0.10,
        RVE_size=[LX,LX,LX], aspect_ratio=5.0,
        orientation_type='axis', orientation_params={'axis': 'y'},
        seed=1234)
    rve.generate_RSA(max_attempts=5000)
    vf = rve.actual_volume_fraction
    print_bounds(vf)

    for N in resolutions:
        N_grid = N + 1
        k0 = (K_FIBER + K_CLAY) / 2.
        print(f"\n  == FEM N={N}  FFT/LR N_grid={N_grid} ==")

        X_full   = make_grid(N_grid)
        fmask_np = rve.get_fiber_mask(X_full)

        print("  [FFT]")
        K_fft, _ = fft_keff(fmask_np, K_FIBER, K_CLAY, N_grid, k0)
        print_row(f"N_grid={N_grid}", "FFT", K_fft)

        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, K_FIBER, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(f"N_grid={N_grid}", "LS-RJF", K_lr, K_fft, nfev, t)

        if HAS_FENICS:
            print("  [FEniCS]")
            K_fen, _, t_fen = run_fenics_direct(rve, resolution=N)
            print_row(f"FEM N={N}", "FEniCS", K_fen, K_fft, t=t_fen)

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 6 -- FIBER CONDUCTIVITY SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def series6_kfiber(kf_list=None, N_grid=N_GRID, N_col=N_COL, arch=ARCH):
    if kf_list is None:
        kf_list = [
            (0.05, 'coir/jute'),
            (0.10, 'sisal'),
            (0.20, 'polyester'),
            (0.29, 'baseline'),
            (0.50, 'PE'),
            (1.00, 'glass-E'),
        ]
    sep("SERIES 6 -- Fiber Conductivity Sweep")
    print(f"  km={K_CLAY} fixed  Vf=10%, alpha=5, O-y  "
          f"Ng={N_grid}  Nc={N_col}  arch={arch}")
    print("  k0 = (kf + km) / 2 adaptive for each kf")

    rve = RVE_Cylinder(
        n_inclusions=20, volume_fraction=0.10,
        RVE_size=[LX,LX,LX], aspect_ratio=5.0,
        orientation_type='axis', orientation_params={'axis': 'y'},
        seed=1234)
    rve.generate_RSA(max_attempts=5000)
    vf = rve.actual_volume_fraction
    print_bounds(vf)

    X_full   = make_grid(N_grid)
    fmask_np = rve.get_fiber_mask(X_full)

    for kf_val, label in kf_list:
        k0 = (kf_val + K_CLAY) / 2.
        print(f"\n  == kf={kf_val} ({label})  k0={k0:.4f} ==")

        print("  [FFT]")
        K_fft, _ = fft_keff(fmask_np, kf_val, K_CLAY, N_grid, k0)
        print_row(f"kf={kf_val} {label}", "FFT", K_fft)
        """
        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, kf_val, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(f"kf={kf_val} {label}", "LS-RJF", K_lr, K_fft, nfev, t)
        """
        if HAS_FENICS:
         print("\n  [FEniCS]")
         K_fen, eps_fen, t_fen = run_fenics_direct(rve, resolution=30,
                                                    kf=kf_val, km=K_CLAY)
         print_tensor(K_fen, "K_eff FEniCS", K_ref=K_fft)
         print_row(f" kf={kf_val} {label}", "FEniCS",
                  K_fen, K_fft, t=t_fen)

        # physics check
        aniso = "Kyy>Kxx" if K_lr[1,1] > K_lr[0,0] else "Kyy<Kxx"
        expected = "Kyy>Kxx" if kf_val < K_CLAY else "Kyy<Kxx"
        print(f"  Anisotropy: {aniso}  expected {expected}  "
              f"{'OK' if aniso==expected else 'FAIL'}  "
              f"(kf/km={kf_val/K_CLAY:.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 7 -- REFERENCE MEDIUM k0 SWEEP
# ══════════════════════════════════════════════════════════════════════════════
def series7_k0(k0_list=None, N_grid=N_GRID, N_col=N_COL, arch=ARCH):
    DK = K_CLAY - K_FIBER
    if k0_list is None:
        k0_list = [
            (K_FIBER,                   'k0=kf'),
            (K_FIBER + 0.1 * DK,        'kf+0.1*Dk'),
            ((K_CLAY + K_FIBER) / 2.,   '(km+kf)/2 [base]'),
            (K_CLAY  - 0.1 * DK,        'km-0.1*Dk'),
            (K_CLAY,                    'k0=km'),
        ]
    sep("SERIES 7 -- Reference Medium k0 Sweep")
    print(f"  kf={K_FIBER}  km={K_CLAY}  Vf=10%, alpha=5, O-y  "
          f"Ng={N_grid}  Nc={N_col}")

    rve = RVE_Cylinder(
        n_inclusions=20, volume_fraction=0.10,
        RVE_size=[LX,LX,LX], aspect_ratio=5.0,
        orientation_type='axis', orientation_params={'axis': 'y'},
        seed=1234)
    rve.generate_RSA(max_attempts=5000)
    vf = rve.actual_volume_fraction
    print_bounds(vf)

    X_full   = make_grid(N_grid)
    fmask_np = rve.get_fiber_mask(X_full)

    for k0v, label in k0_list:
        print(f"\n  == k0={k0v:.4f} ({label}) ==")

        print("  [FFT]")
        K_fft, _ = fft_keff(fmask_np, K_FIBER, K_CLAY, N_grid, k0v)
        print_row(f"k0={k0v:.3f}", "FFT", K_fft)

        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, K_FIBER, K_CLAY, k0v, arch=arch, N_col=N_col)
        print_row(f"k0={k0v:.3f} {label}", "LS-RJF", K_lr, K_fft, nfev, t)

# ══════════════════════════════════════════════════════════════════════════════
# SERIES 8 -- HYPERPARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
def series8_hyperparams():
    sep("SERIES 8 -- LS-RJF Hyperparameter Sensitivity")
    print(f"  Baseline RVE: Vf=10%, alpha=5, O-y  "
          f"kf={K_FIBER}  km={K_CLAY}")
    print("  FFT reference fixed at Ng=31")
    print("  One parameter varied at a time; others at baseline")

    rve = RVE_Cylinder(
        n_inclusions=20, volume_fraction=0.10,
        RVE_size=[LX,LX,LX], aspect_ratio=5.0,
        orientation_type='axis', orientation_params={'axis': 'y'},
        seed=1234)
    rve.generate_RSA(max_attempts=5000)
    vf = rve.actual_volume_fraction
    k0 = (K_FIBER + K_CLAY) / 2.
    print_bounds(vf)

    # FFT reference at Ng=31
    X31     = make_grid(31)
    fmask31 = rve.get_fiber_mask(X31)
    print(f"\n  [FFT reference  Ng=31  k0={k0:.4f}]")
    K_fft_ref, _ = fft_keff(fmask31, K_FIBER, K_CLAY, 31, k0)
    print_row("FFT Ng=31", "FFT", K_fft_ref)

    def run_one(cfg, N_grid, N_col, arch):
        X_f  = make_grid(N_grid)
        fm   = rve.get_fiber_mask(X_f)
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_f, fm, K_FIBER, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(cfg, "LS-RJF", K_lr, K_fft_ref, nfev, t)


    # 8b: Nc sweep
    print("\n  8b -- Collocation Nc  (Ng=31, arch=[6,20,20,20,1])")
    for nc in [100, 200, 500]:
        base = " (base)" if nc == 500 else ""
        run_one(f"Nc={nc}{base}", 31, nc, (6,20,20,1))

    # 8c: n_layers sweep
    print("\n  8c -- Hidden layers  (Ng=31, Nc=200, width=20 per layer)")
    for n_layers in [1, 2, 3, 4]:
        arch = tuple([6] + [20] * n_layers + [1])
        base = " (base)" if n_layers == 2 else ""
        run_one(f"arch={list(arch)}{base}", 31, 500, arch)


def series9_ninclusions(n_list=None, N_grid=N_GRID, N_col=N_COL,
                     FEM_RES=30, arch=ARCH):
    if n_list is None:
        n_list = [10,20,30,40,50]
    sep("SERIES 9 -- Number of Inclusions Sweep")
    print(f"  n list: {n_list}  alpha=5, O-y  "
          f"Ng={N_grid}  Nc={N_col}  arch={arch}")

    for n_inc in n_list:
        print(f"\n  == n_inc={n_inc} ==")
        rve = RVE_Cylinder(
            n_inclusions=n_inc, volume_fraction=0.1,
            RVE_size=[LX,LX,LX], aspect_ratio=5.0,
            orientation_type='axis', orientation_params={'axis': 'y'},
            seed=1234)
        rve.generate_RSA(max_attempts=8000)
        vf = rve.actual_volume_fraction
        k0 = (K_FIBER + K_CLAY) / 2.
        voigt, reuss, mg = print_bounds(vf)

        X_full   = make_grid(N_grid)
        fmask_np = rve.get_fiber_mask(X_full)

        print("  [FFT]")
        K_fft, _, = fft_keff(fmask_np, K_FIBER, K_CLAY, N_grid, k0)
        print_row(f"Vf={vf:.3f}", "FFT", K_fft)

        print("  [LS-RJF]")
        K_lr, _, nfev, t = run_lsrjf_direct(
            X_full, fmask_np, K_FIBER, K_CLAY, k0, arch=arch, N_col=N_col)
        print_row(f"Vf={vf:.3f}", "LS-RJF", K_lr, K_fft, nfev, t)

        if HAS_FENICS:
            print("  [FEniCS]")
            K_fen, _, t_fen = run_fenics_direct(rve, resolution=FEM_RES)
            print_row(f"Vf={vf:.3f}", "FEniCS", K_fen, K_fft, t=t_fen)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sep("LS-RJF DIRECT -- All Series")
    print(f"  kf={K_FIBER}  km={K_CLAY}  k0={K0:.4f}")
    print(f"  arch={ARCH}  N_col={N_COL}  N_grid={N_GRID}")
    sep()

    #series1_baseline()

    #series2_vf_sweep()
    #series3_ar_sweep()
    #series4_orientation()
    #series5_resolution()
    #series6_kfiber()
    #series7_k0()
    #series8_hyperparams()
    series9_ninclusions()

    sep("DONE")
