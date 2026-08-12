"""
RVE_Cylinder — Representative Volume Element for thermal homogenization.
  get_fiber_mask_tf : native TensorFlow version of the fiber/matrix
                      mask, to stay inside the differentiable graph
                      without ever going through .numpy().
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

# Default thermal properties
k_matrix = 0.651   # W/(m*K)  matrix (clay)
k_fiber  = 0.29    # W/(m*K)  fiber

# Accepted orientation types
VALID_ORIENTATION_TYPES = {'random', 'axis', 'plane', 'preferred'}


# ============================================================================
# Orientation
# ============================================================================
def generate_orientation_matrix(orientation_type='random', params=None, rng=None):
    """
    Generates a 3x3 rotation matrix whose 3rd column is the cylinder's
    principal axis.

    Parameters
    ----------
    orientation_type : 'random' | 'axis' | 'plane' | 'preferred'
    params : dict
        - 'axis'      -> {'axis': 'x'|'y'|'z'}
        - 'plane'     -> {'normal': [nx, ny, nz]}
        - 'preferred' -> {'preferred': [px, py, pz], 'noise': float}
    rng : np.random.Generator, optional

    FIX-ORI: raises ValueError for any unknown orientation_type.
    """
    if params is None:
        params = {}
    if rng is None:
        rng = np.random.default_rng()

    if orientation_type not in VALID_ORIENTATION_TYPES:
        raise ValueError(
            f"orientation_type='{orientation_type}' invalid. "
            f"Accepted values: {sorted(VALID_ORIENTATION_TYPES)}.\n"
            f"  'random'    : uniformly random orientation\n"
            f"  'axis'      : aligned on x, y or z  (params={{'axis':'x'|'y'|'z'}})\n"
            f"  'plane'     : confined to a plane  (params={{'normal':[nx,ny,nz]}})\n"
            f"  'preferred' : biased toward a direction "
            f"(params={{'preferred':[px,py,pz], 'noise':float}})"
        )

    if orientation_type == 'axis':
        axis_name = params.get('axis', 'z').lower()
        if axis_name not in {'x', 'y', 'z'}:
            raise ValueError(
                f"axis='{axis_name}' invalid for orientation_type='axis'. "
                f"Must be 'x', 'y' or 'z'."
            )
        v = {'x': np.array([1., 0., 0.]),
             'y': np.array([0., 1., 0.]),
             'z': np.array([0., 0., 1.])}[axis_name]

    elif orientation_type == 'plane':
        normal = np.array(params.get('normal', [0., 0., 1.]), dtype=float)
        normal /= np.linalg.norm(normal) + 1e-8
        v = rng.standard_normal(3)
        v -= np.dot(v, normal) * normal
        v /= np.linalg.norm(v) + 1e-8
        if 'preferred' in params:
            pref = np.array(params['preferred'], dtype=float)
            pref -= np.dot(pref, normal) * normal
            if np.linalg.norm(pref) > 1e-8:
                pref /= np.linalg.norm(pref)
                alpha = params.get('bias_strength', 0.7)
                v = (1 - alpha) * v + alpha * pref
                v /= np.linalg.norm(v) + 1e-8

    elif orientation_type == 'preferred':
        pref = np.array(params.get('preferred', [1., 0., 0.]), dtype=float)
        pref /= np.linalg.norm(pref) + 1e-8
        noise = params.get('noise', 0.2)
        v = pref + rng.standard_normal(3) * noise
        v /= np.linalg.norm(v) + 1e-8

    else:  # 'random'
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v) + 1e-8

    # Orthonormal frame {v2, v3, v}
    temp = np.array([1., 0., 0.]) if abs(v[0]) < 0.9 else np.array([0., 1., 0.])
    v2 = np.cross(v, temp);  v2 /= np.linalg.norm(v2) + 1e-8
    v3 = np.cross(v, v2);    v3 /= np.linalg.norm(v3) + 1e-8
    return np.column_stack([v2, v3, v])


# ============================================================================
# Cylinder (inclusion)
# ============================================================================
class Cylinder:
    """
    A right circular cylinder defined by:
      center      : geometric center (3,)
      radius      : radius
      length      : total length (= 2 * half-length)
      orientation : 3x3 matrix whose 3rd column is the cylinder axis
    """

    def __init__(self, center, radius, length, orientation=None):
        self.center      = np.array(center, dtype=float)
        self.radius      = float(radius)
        self.length      = float(length)
        self.orientation = orientation if orientation is not None else np.eye(3)
        # Convenience shortcuts
        self.dimensions  = np.array([self.radius, self.length])

    # -- Volume ---------------------------------------------------------
    def get_volume(self):
        return np.pi * self.radius**2 * self.length

    # -- Signed distance to the surface ----------------------------------
    def distance_to_point(self, point):
        """
        Signed distance (< 0 inside, > 0 outside).
        """
        local   = self.orientation.T @ (point - self.center)
        d_rad   = np.sqrt(local[0]**2 + local[1]**2) - self.radius
        d_ax    = abs(local[2]) - self.length / 2
        if d_ax > 0 and d_rad > 0:
            return np.sqrt(d_rad**2 + d_ax**2)
        elif d_ax > 0:
            return d_ax
        elif d_rad > 0:
            return d_rad
        else:
            return max(d_rad, d_ax)

    def is_inside(self, point):
        return self.distance_to_point(point) < 0

    # -- Outward normal ---------------------------------------------------
    def outward_normal(self, point):
        """
        Unit outward normal at a point on the surface.

        FIX-CYL + FIX-SIGN:
          - Lateral vs endcap is detected via |d_axial| vs |d_radial|.
          - The left endcap correctly returns -axis (sign(local[2]) * axis).
        """
        r, l  = self.radius, self.length
        axis  = self.orientation[:, 2]
        local = self.orientation.T @ (point - self.center)

        d_rad = np.sqrt(local[0]**2 + local[1]**2) - r
        d_ax  = abs(local[2]) - l / 2

        if abs(d_ax) < abs(d_rad):
            # -- Endcap (FIX-CYL + FIX-SIGN) --------------------------
            return np.sign(local[2]) * axis
        else:
            # -- Lateral surface -----------------------------------------
            v    = point - self.center
            perp = v - np.dot(v, axis) * axis
            nm   = np.linalg.norm(perp)
            return perp / (nm + 1e-8) if nm > 1e-8 else axis


# ============================================================================
# Intersection detection between cylinders
# ============================================================================
class CylinderIntersectionTester:

    @staticmethod
    def _point_inside(point, cyl, tol=1e-6):
        v    = point - cyl.center
        a    = cyl.orientation[:, 2]
        proj = np.dot(v, a)
        if abs(proj) > cyl.length / 2 + tol:
            return False
        return np.linalg.norm(v - proj * a) <= cyl.radius + tol

    @staticmethod
    def overlap(c1, c2, tol=1e-6):
        """Tests whether two cylinders overlap."""
        r1, l1 = c1.radius, c1.length
        r2, l2 = c2.radius, c2.length
        p1, p2 = c1.center, c2.center
        a1     = c1.orientation[:, 2]
        a2     = c2.orientation[:, 2]
        v      = p1 - p2
        dot    = np.dot(a1, a2)
        det    = 1 - dot * dot

        if abs(abs(dot) - 1.0) < 1e-6:
            # (Quasi-)parallel cylinders
            proj   = np.dot(v, a1)
            d_perp = np.linalg.norm(v - proj * a1)
            if d_perp > r1 + r2 - tol:
                return False
            overlap_len = max(0,
                min(proj + l1/2,  l2/2) -
                max(proj - l1/2, -l2/2))
            if overlap_len > 0:
                return True
            for sign in (-1, 1):
                if CylinderIntersectionTester._point_inside(p1 + sign*(l1/2)*a1, c2, tol):
                    return True
                if CylinderIntersectionTester._point_inside(p2 + sign*(l2/2)*a2, c1, tol):
                    return True
            return False
        else:
            # Non-parallel cylinders -- find the closest point
            rhs1  = -np.dot(v, a1)
            rhs2  = -np.dot(v, a2)
            u     = (rhs1 - dot * rhs2) / det
            v_val = (dot * rhs1 - rhs2) / det
            pt1   = p1 + u * a1
            pt2   = p2 + v_val * a2
            in1   = abs(u)     <= l1/2 + tol
            in2   = abs(v_val) <= l2/2 + tol
            if in1 and in2:
                return np.linalg.norm(pt1 - pt2) < r1 + r2 - tol
            for sign in (-1, 1):
                if CylinderIntersectionTester._point_inside(p1 + sign*(l1/2)*a1, c2, tol):
                    return True
                if CylinderIntersectionTester._point_inside(p2 + sign*(l2/2)*a2, c1, tol):
                    return True
            return False


# ============================================================================
# RVE -- oriented cylinders
# ============================================================================
class RVE_Cylinder:
    """
    Periodic representative volume element composed of oriented cylinders.

    Parameters
    ----------
    n_inclusions    : number of cylinders to place
    volume_fraction : target volume fraction
    RVE_size        : [Lx, Ly, Lz]  cell dimensions
    aspect_ratio    : l / (2r)  length/diameter ratio  (default 5)
    orientation_type: 'random' | 'axis' | 'plane' | 'preferred'
    orientation_params : dict passed to generate_orientation_matrix()
    material_props  : {'k_matrix': float, 'k_fiber': float}
    seed            : RNG seed
    """

    def __init__(self, n_inclusions, volume_fraction, RVE_size,
                 aspect_ratio=5.0,
                 orientation_type='random', orientation_params=None,
                 material_props=None, seed=None):

        # FIX-ORI: immediate validation
        if orientation_type not in VALID_ORIENTATION_TYPES:
            raise ValueError(
                f"orientation_type='{orientation_type}' invalid. "
                f"Accepted: {sorted(VALID_ORIENTATION_TYPES)}."
            )

        self.n_inclusions       = n_inclusions
        self.volume_fraction    = volume_fraction
        self.RVE_size           = np.array(RVE_size, dtype=float)
        self.aspect_ratio       = float(aspect_ratio)
        self.orientation_type   = orientation_type
        self.orientation_params = orientation_params or {}
        self._rng               = np.random.default_rng(seed)

        mp = material_props or {}
        self.k_matrix = mp.get('k_matrix', k_matrix)
        self.k_fiber  = mp.get('k_fiber',  k_fiber)

        self._compute_dimensions()
        self.cylinders              = []
        self.actual_volume_fraction = 0.0
        self.periodic_faces         = None

    # -- Computing a cylinder's dimensions ---------------------------------
    def _compute_dimensions(self):
        total_vol   = self.volume_fraction * np.prod(self.RVE_size)
        vol_per_cyl = total_vol / self.n_inclusions
        # V = pi r^2 l = pi r^2 (2 * aspect_ratio * r) = 2*pi * aspect_ratio * r^3
        self.radius = (vol_per_cyl / (2 * np.pi * self.aspect_ratio)) ** (1/3)
        self.length = 2 * self.aspect_ratio * self.radius
        print(f"  Cylinder: r={self.radius:.4f}, l={self.length:.4f}, "
              f"AR={self.aspect_ratio}")

    # -- Orientation ---------------------------------------------------------
    def _generate_orientation(self):
        return generate_orientation_matrix(
            self.orientation_type, self.orientation_params, rng=self._rng)

    # -- Periodicity ---------------------------------------------------------
    def _periodic_vector(self, p1, p2):
        delta = p1 - p2
        for i in range(3):
            L = self.RVE_size[i]
            if L > 0:
                delta[i] -= np.round(delta[i] / L) * L
        return delta

    def _periodic_distance(self, p1, p2):
        return np.linalg.norm(self._periodic_vector(p1, p2))

    def _periodic_image(self, point, reference):
        """Returns the periodic image of `point` closest to `reference`."""
        return reference + self._periodic_vector(point, reference)

    # -- Phase -----------------------------------------------------------------
    def _is_inside(self, point, cyl):
        p_img = self._periodic_image(point, cyl.center)
        return cyl.is_inside(p_img)

    def is_point_in_fiber(self, point):
        return any(self._is_inside(point, cyl) for cyl in self.cylinders)

    def get_conductivity(self, point):
        return self.k_fiber if self.is_point_in_fiber(point) else self.k_matrix

    def get_fiber_mask(self, points):
        """Returns a boolean mask for an (N, 3) array of points."""
        points = np.asarray(points)
        mask   = np.zeros(len(points), dtype=bool)
        for cyl in self.cylinders:
            for i, p in enumerate(points):
                if not mask[i] and self._is_inside(p, cyl):
                    mask[i] = True
        return mask

    # -- Native TensorFlow version (ADDED) ------------------------------------
    def get_fiber_mask_tf(self, points_tf, RVE_size_tf=None):
        """
        Native TensorFlow version of get_fiber_mask, vectorized over all
        points and all cylinders, WITH PERIODICITY TAKEN INTO ACCOUNT
        (unlike a naive version which would omit it).

        points_tf : Tensor (N,3), float32
        Returns   : Tensor (N,), float32 (1.0 = fiber, 0.0 = matrix)

        Stays inside the TF graph -- no .numpy(), usable in a residual
        differentiated by tape.jacobian (the mask itself has no useful
        gradient here since the geometry is fixed, but it does not
        break the trace for downstream variables such as kf).
        """
        import tensorflow as tf
        points_tf = tf.cast(points_tf, tf.float32)
        L = tf.constant(self.RVE_size, dtype=tf.float32) if RVE_size_tf is None \
            else RVE_size_tf
        N = tf.shape(points_tf)[0]
        mask = tf.zeros([N], dtype=tf.bool)

        for cyl in self.cylinders:
            center = tf.constant(cyl.center, dtype=tf.float32)
            orient = tf.constant(cyl.orientation, dtype=tf.float32)

            # Closest periodic image: delta = p - center, wrapped into
            # [-L/2, L/2] component-wise.
            delta = points_tf - center
            delta = delta - L * tf.round(delta / L)
            local = tf.matmul(delta, orient)   # (N,3) in the cylinder's frame

            d_rad = tf.sqrt(local[:,0]**2 + local[:,1]**2) - cyl.radius
            d_ax  = tf.abs(local[:,2]) - cyl.length/2
            inside = tf.logical_and(d_rad < 0, d_ax < 0)
            mask = tf.logical_or(mask, inside)

        return tf.cast(mask, tf.float32)

    # -- Periodic overlap ------------------------------------------------------
    def _overlap_periodic(self, c1, c2, tol=1e-6):
        if CylinderIntersectionTester.overlap(c1, c2, tol):
            return True
        delta  = self._periodic_vector(c1.center, c2.center)
        c2_img = Cylinder(c1.center - delta, c2.radius, c2.length, c2.orientation)
        return CylinderIntersectionTester.overlap(c1, c2_img, tol)

    # -- RSA ---------------------------------------------------------------------
    def generate_RSA(self, max_attempts=5000, tol=1e-6):
        """Random Sequential Adsorption placement."""
        self.cylinders = []
        total_vol      = 0.0
        attempts       = 0
        placed         = 0

        print(f"Periodic RSA: {self.n_inclusions} cylinders, "
              f"orientation='{self.orientation_type}'")

        while placed < self.n_inclusions and attempts < max_attempts:
            attempts += 1
            center      = self._rng.random(3) * self.RVE_size
            orientation = self._generate_orientation()
            cyl         = Cylinder(center, self.radius, self.length, orientation)

            if not any(self._overlap_periodic(cyl, ex, tol) for ex in self.cylinders):
                self.cylinders.append(cyl)
                placed    += 1
                total_vol += cyl.get_volume()
                if placed % 10 == 0:
                    vf = total_vol / np.prod(self.RVE_size)
                    print(f"  {placed} cylinders, Vf={vf:.4f}")

        self.actual_volume_fraction = total_vol / np.prod(self.RVE_size)
        print(f"RSA done: {placed} cylinders, "
              f"Vf={self.actual_volume_fraction:.4f}")
        if placed < self.n_inclusions:
            print(f"  Warning: only {placed}/{self.n_inclusions} placed "
                  f"(increase max_attempts or reduce volume_fraction)")

    # -- Interface sampling --------------------------------------------------
    def sample_interface_points(self, n_points=2000, eps=1e-3):
        """
        Draws random points and keeps those close to a cylinder surface.
        eps is automatically adapted to the inclusion size.
        """
        if self.cylinders:
            min_dim = min(min(c.radius, c.length) for c in self.cylinders)
            eps     = max(eps, 0.01 * min_dim)

        points, indices = [], []
        candidates = self._rng.uniform(0, self.RVE_size,
                                       size=(n_points * 10, 3))
        for p in candidates:
            for i, cyl in enumerate(self.cylinders):
                p_img = self._periodic_image(p, cyl.center)
                if abs(cyl.distance_to_point(p_img)) < eps:
                    points.append(p)
                    indices.append(i)
                    break
            if len(points) >= n_points:
                break

        if not points:
            print(f"  Warning: no interface point found with eps={eps:.2e}.")
        return np.array(points, dtype=float), np.array(indices, dtype=int)

    def sample_interface_points_with_normals(self, n_points=1000, eps=1e-3):
        pts, indices = self.sample_interface_points(n_points, eps)
        normals = []
        for p, idx in zip(pts, indices):
            cyl   = self.cylinders[idx]
            p_img = self._periodic_image(p, cyl.center)
            normals.append(cyl.outward_normal(p_img))
        return (np.array(pts,     dtype=np.float32),
                np.array(normals, dtype=np.float32))

    # -- Normal verification -----------------------------------------------------
    def _valid_mask(self, pts, norms, eps=1e-2):
        valid = np.ones(len(pts), dtype=bool)

        # 1. Unit normals
        mags = np.linalg.norm(norms, axis=1)
        valid[np.abs(mags - 1.0) > 1e-4] = False

        # 2. Lateral surface -> weak axial component
        for cyl in self.cylinders:
            axis = cyl.orientation[:, 2]
            for i, (p, n) in enumerate(zip(pts, norms)):
                if not valid[i]:
                    continue
                p_img = self._periodic_image(p, cyl.center)
                if abs(cyl.distance_to_point(p_img)) > eps:
                    continue
                local = cyl.orientation.T @ (p_img - cyl.center)
                d_rad = np.sqrt(local[0]**2 + local[1]**2) - cyl.radius
                d_ax  = abs(local[2]) - cyl.length / 2
                if abs(d_rad) < abs(d_ax):   # lateral surface
                    if abs(np.dot(n, axis)) > 0.1:
                        valid[i] = False
        return valid

    def verify_interface_normals_from(self, pts, norms, eps=1e-2):
        if len(pts) == 0:
            print("  Warning: no interface points")
            return False
        valid = self._valid_mask(pts, norms, eps)
        n_bad = np.sum(~valid)
        if n_bad:
            print(f"  FAIL: {n_bad} invalid normals out of {len(pts)}")
            return False
        print(f"  OK: {len(pts)} correct interface normals")
        return True

    def get_valid_interface_points(self, pts, norms, eps=1e-2):
        if len(pts) == 0:
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32))
        valid = self._valid_mask(pts, norms, eps)
        return pts[valid].astype(np.float32), norms[valid].astype(np.float32)

    # -- Periodic boundary conditions ----------------------------------------
    def generate_faces(self, n_per_axis=30):
        Lx, Ly, Lz = self.RVE_size
        self.periodic_faces = {}
        y = np.linspace(0, Ly, n_per_axis)
        z = np.linspace(0, Lz, n_per_axis)
        x = np.linspace(0, Lx, n_per_axis)
        yg, zg = np.meshgrid(y, z, indexing='ij')
        yf, zf = yg.flatten(), zg.flatten()
        self.periodic_faces['x0'] = np.column_stack([np.zeros_like(yf),    yf, zf])
        self.periodic_faces['xL'] = np.column_stack([Lx*np.ones_like(yf),  yf, zf])
        xg, zg = np.meshgrid(x, z, indexing='ij')
        xf, zf = xg.flatten(), zg.flatten()
        self.periodic_faces['y0'] = np.column_stack([xf, np.zeros_like(xf),    zf])
        self.periodic_faces['yL'] = np.column_stack([xf, Ly*np.ones_like(xf),  zf])
        xg, yg = np.meshgrid(x, y, indexing='ij')
        xf, yf = xg.flatten(), yg.flatten()
        self.periodic_faces['z0'] = np.column_stack([xf, yf, np.zeros_like(xf)])
        self.periodic_faces['zL'] = np.column_stack([xf, yf, Lz*np.ones_like(xf)])
        print(f"Faces generated: {len(self.periodic_faces['x0'])} pts/face")

    # -- Checks -------------------------------------------------------------
    def verify_no_overlap(self, tol=1e-6):
        n = len(self.cylinders)
        overlaps = sum(
            1 for i in range(n) for j in range(i+1, n)
            if self._overlap_periodic(self.cylinders[i], self.cylinders[j], tol)
        )
        if overlaps == 0:
            print("OK: no overlap")
            return True
        print(f"FAIL: {overlaps} overlap(s) detected")
        return False

    def verify_periodicity(self, n_samples=500):
        print("Checking periodicity...")
        errors = 0
        rng    = np.random.default_rng(0)
        for _ in range(n_samples):
            p          = rng.random(3) * self.RVE_size
            phase_orig = self.is_point_in_fiber(p)
            for axis in range(3):
                p_shift        = p.copy()
                p_shift[axis] += self.RVE_size[axis]
                p_shift        = p_shift % self.RVE_size
                if phase_orig != self.is_point_in_fiber(p_shift):
                    errors += 1
        if errors == 0:
            print("OK: periodicity verified")
            return True
        print(f"FAIL: {errors} periodicity error(s)")
        return False

    # -- 3D plot --------------------------------------------------------------
    def plot_3d(self, figsize=(10, 8)):
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')

        Lx, Ly, Lz = self.RVE_size
        verts = np.array([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0],
                          [0,0,Lz],[Lx,0,Lz],[Lx,Ly,Lz],[0,Ly,Lz]])
        edges = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        for e in edges:
            ax.plot3D(*zip(verts[e[0]], verts[e[1]]), 'k-', alpha=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.cylinders)))
        for cyl, col in zip(self.cylinders, colors):
            self._plot_cylinder(ax, cyl, col)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(
            f'RVE Cylinders -- {len(self.cylinders)} inclusions, '
            f"orientation='{self.orientation_type}', "
            f'Vf={self.actual_volume_fraction:.3f}')
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
        plt.tight_layout()
        return fig, ax

    def _plot_cylinder(self, ax, cyl, color):
        """
        FIX-PLOT: uses inc.orientation[:, 0/1/2] directly.
        Cylinders are drawn according to their actual orientation.
        """
        r, l   = cyl.radius, cyl.length
        axis   = cyl.orientation[:, 2]   # axis      <- FIX-PLOT
        perp1  = cyl.orientation[:, 0]   # radial 1  <- FIX-PLOT
        perp2  = cyl.orientation[:, 1]   # radial 2  <- FIX-PLOT

        theta  = np.linspace(0, 2*np.pi, 30)
        z_vals = np.linspace(-l/2, l/2,  10)
        tg, zg = np.meshgrid(theta, z_vals)

        # Lateral surface
        pts = (r * np.cos(tg)[:,:,None] * perp1
             + r * np.sin(tg)[:,:,None] * perp2
             + zg[:,:,None]             * axis
             + cyl.center)
        ax.plot_surface(pts[:,:,0], pts[:,:,1], pts[:,:,2],
                        color=color, alpha=0.6, linewidth=0)

        # Endcaps
        rr = np.linspace(0, r, 8)
        rg, tg2 = np.meshgrid(rr, theta)
        for sign in (-1, 1):
            cap_center = cyl.center + sign * (l/2) * axis
            cap = (rg[:,:,None] * np.cos(tg2)[:,:,None] * perp1
                 + rg[:,:,None] * np.sin(tg2)[:,:,None] * perp2
                 + cap_center)
            ax.plot_surface(cap[:,:,0], cap[:,:,1], cap[:,:,2],
                            color=color, alpha=0.6, linewidth=0)


# ============================================================================
# TEST SUITE
# ============================================================================
def run_tests():
    print("=" * 60)
    print("RVE_Cylinder TESTS")
    print("=" * 60)
    all_pass = True

    # -- T0: orientation validation ------------------------------------------
    print("\n[T0] orientation_type validation (FIX-ORI)")
    for bad in ['x', 'y', 'z', 'axial', 'none', 'X']:
        try:
            generate_orientation_matrix(bad)
            print(f"  FAIL: '{bad}' should have raised ValueError")
            all_pass = False
        except ValueError:
            print(f"  OK: '{bad}' -> ValueError correctly raised")

    for otype, params in [
        ('random',    {}),
        ('axis',      {'axis': 'x'}),
        ('axis',      {'axis': 'y'}),
        ('axis',      {'axis': 'z'}),
        ('plane',     {'normal': [0, 0, 1]}),
        ('preferred', {'preferred': [1, 0, 0], 'noise': 0.1}),
    ]:
        M   = generate_orientation_matrix(otype, params)
        err = np.max(np.abs(M.T @ M - np.eye(3)))
        ok  = err < 1e-6
        print(f"  {'OK' if ok else 'FAIL'} '{otype}' {params} -> "
              f"orthonormal={'yes' if ok else 'NO'} (err={err:.1e})")
        if not ok: all_pass = False

    # -- T1: lateral surface normals ------------------------------------------
    print("\n[T1] Lateral surface normals (axis=x)")
    orient = generate_orientation_matrix('axis', {'axis': 'x'})
    cyl    = Cylinder([0.5, 0.5, 0.5], 0.08, 0.6, orient)
    axis   = orient[:, 2]
    lat_pts = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        theta = rng.uniform(0, 2*np.pi)
        t     = rng.uniform(-0.25, 0.25)
        eps   = rng.uniform(-0.004, 0.004)
        lat_pts.append(
            cyl.center
            + (0.08 + eps) * (np.cos(theta)*np.array([0,1,0])
                             + np.sin(theta)*np.array([0,0,1]))
            + t * axis)
    n_lat  = np.array([cyl.outward_normal(p) for p in lat_pts])
    mags   = np.linalg.norm(n_lat, axis=1)
    ax_comp = np.abs(n_lat @ axis)
    ok = np.allclose(mags, 1.0, atol=1e-5) and np.all(ax_comp < 0.05)
    print(f"  Unit norm : {'OK' if np.allclose(mags,1,atol=1e-5) else 'FAIL'}")
    print(f"  Perp to axis : {'OK' if np.all(ax_comp<0.05) else 'FAIL'} "
          f"(max={ax_comp.max():.4f})")
    if not ok: all_pass = False

    # -- T2: endcap normals (FIX-CYL + FIX-SIGN) -------------------------------
    print("\n[T2] Endcap normals (FIX-CYL + FIX-SIGN)")
    right_pts, left_pts = [], []
    for _ in range(100):
        rr    = rng.uniform(0, 0.06)
        theta = rng.uniform(0, 2*np.pi)
        off   = rr*(np.cos(theta)*np.array([0,1,0])
                   + np.sin(theta)*np.array([0,0,1]))
        eps   = rng.uniform(0.002, 0.008)
        right_pts.append(cyl.center + (0.3 + eps)*axis + off)
        left_pts.append( cyl.center - (0.3 + eps)*axis + off)
    n_right   = np.array([cyl.outward_normal(p) for p in right_pts])
    n_left    = np.array([cyl.outward_normal(p) for p in left_pts])
    dot_right = n_right @ axis
    dot_left  = n_left  @ axis
    ok_r      = np.all(dot_right >  0.98)
    ok_l      = np.all(dot_left  < -0.98)
    print(f"  Right endcap  (+axis) : {'OK' if ok_r else 'FAIL'} "
          f"(min dot={dot_right.min():.4f})")
    print(f"  Left endcap (-axis) : {'OK' if ok_l else 'FAIL'} "
          f"(max dot={dot_left.max():.4f})")
    if not (ok_r and ok_l): all_pass = False

    # -- T3-T7: full RVE for different orientations --------------------------
    configs = [
        ('random',    {},              0.10, 15),
        ('axis',      {'axis': 'x'},   0.10, 15),
        ('axis',      {'axis': 'y'},   0.10, 15),
        ('axis',      {'axis': 'z'},   0.10, 15),
        ('preferred', {'preferred': [1,1,0], 'noise': 0.3}, 0.08, 12),
    ]
    for k, (otype, oparams, vf, n) in enumerate(configs, start=3):
        print(f"\n[T{k}] Cylinder RVE -- orientation='{otype}' {oparams}")
        rve = RVE_Cylinder(n_inclusions=n, volume_fraction=vf,
                           RVE_size=[1, 1, 1],
                           orientation_type=otype,
                           orientation_params=oparams,
                           seed=42)
        rve.generate_RSA()
        ok1 = rve.verify_no_overlap()
        ok2 = rve.verify_periodicity(n_samples=300)
        pts, norms = rve.sample_interface_points_with_normals(n_points=800)
        ok3 = rve.verify_interface_normals_from(pts, norms)
        pts_v, norms_v = rve.get_valid_interface_points(pts, norms)
        print(f"  Valid interface points: {len(pts_v)}/{len(pts)}")
        if not (ok1 and ok2 and ok3): all_pass = False

    # -- T8: axis vector consistent with orientation_type='axis' -------------
    print("\n[T8] Consistency of orientation[:, 2] for type='axis' (FIX-PLOT)")
    expected = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
    for ax_name, exp in expected.items():
        M   = generate_orientation_matrix('axis', {'axis': ax_name})
        dot = abs(np.dot(M[:, 2], exp))
        ok  = dot > 0.999
        print(f"  axis='{ax_name}' -> orientation[:,2].e_{ax_name} = {dot:.6f} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok: all_pass = False

    print("\n" + "="*60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("="*60)
    return all_pass


if __name__ == "__main__":
    run_tests()
