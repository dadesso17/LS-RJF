"""
plot_direct_series.py
=====================
Figures pour l'article LS-RJF direct.
Baseline: Nc=500, arch=[6,20,20,1].
Génère fig_series2_vf.pdf/png, fig_series3_alpha.pdf/png,
fig_series4_orientation.pdf/png, fig_series5_resolution.pdf/png,
fig_series6_kfiber.pdf/png, fig_series7_k0.pdf/png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'lines.linewidth': 2,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150,
})

C_FFT = '#1f77b4'; C_LR = '#d62728'; C_FEM = '#2ca02c'
W = 0.25  # bar width

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{name}.pdf', dpi=150); fig.savefig(f'{name}.png', dpi=150)
    print(f'  Saved: {name}.pdf/png')
    plt.close(fig)

# ============================================================
# SERIES 2 -- Volume fraction
# ============================================================
vf_list = [5, 10, 15, 20, 25]
Kxx_fft = [0.62713, 0.60336, 0.57892, 0.56009, 0.53705]
Kyy_fft = [0.63254, 0.61299, 0.59334, 0.57625, 0.55516]
Kzz_fft = [0.62721, 0.60321, 0.58079, 0.55968, 0.53530]
Kxx_lr  = [0.62984, 0.60660, 0.58297, 0.56404, 0.54088]
Kyy_lr  = [0.63294, 0.61358, 0.59418, 0.57776, 0.55745]
Kzz_lr  = [0.62937, 0.60677, 0.58413, 0.56396, 0.53975]
Kxx_fem = [0.62749, 0.60490, 0.58379, 0.56281, 0.54080]
Kyy_fem = [0.63149, 0.61239, 0.59445, 0.57597, 0.55625]
Kzz_fem = [0.62769, 0.60498, 0.58407, 0.56335, 0.54185]
emax_lr  = [0.43, 0.59, 0.70, 0.76, 0.83]
emax_fem = [0.17, 0.29, 0.84, 0.65, 1.22]
# Voigt and MG bounds
kf=0.29; km=0.651
vf_arr = np.array(vf_list)/100
voigt = vf_arr*kf + (1-vf_arr)*km
mg    = km*(kf+km+vf_arr*(kf-km))/(kf+km-vf_arr*(kf-km))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.plot(vf_list, Kyy_fft, 'o-', color=C_FFT, label=r'$K_{yy}$ FFT')
ax.plot(vf_list, Kxx_fft, 's--', color=C_FFT, label=r'$K_{xx}$ FFT', alpha=0.6)
ax.plot(vf_list, Kyy_lr,  'o-', color=C_LR,  label=r'$K_{yy}$ LS-RJF')
ax.plot(vf_list, Kxx_lr,  's--', color=C_LR,  label=r'$K_{xx}$ LS-RJF', alpha=0.6)
ax.plot(vf_list, Kyy_fem, 'o-', color=C_FEM, label=r'$K_{yy}$ FEM')
ax.plot(vf_list, Kxx_fem, 's--', color=C_FEM, label=r'$K_{xx}$ FEM', alpha=0.6)
ax.plot(vf_list, voigt, 'k--', lw=1.5, label='Voigt')
ax.plot(vf_list, mg,    'k:',  lw=1.5, label='MG')
ax.set_xlabel(r'$v_f$ (%)')
ax.set_ylabel(r'$K$ [W/(m$\cdot$K)]')
ax.set_title('(a) Diagonal components')
ax.legend(fontsize=7, ncol=2)

ax = axes[1]
Kxy_fft = [0.00006, 0.00001, 0.00002, -0.00007, -0.00006]
Kxy_lr  = [0.00004, 0.00005, 0.00011,  0.00003, -0.00010]
ax.plot(vf_list, np.abs(Kxy_fft), 'o-', color=C_FFT, label='|Kxy| FFT')
ax.plot(vf_list, np.abs(Kxy_lr),  'o-', color=C_LR,  label='|Kxy| LS-RJF')
ax.set_xlabel(r'$v_f$ (%)'); ax.set_ylabel(r'$|K_{ij}|$ [W/(m$\cdot$K)]')
ax.set_title('(b) Off-diagonal magnitude'); ax.legend()
ax.set_ylim(bottom=0)

ax = axes[2]
ax.plot(vf_list, emax_lr,  'o-', color=C_LR,  label='LS-RJF')
ax.plot(vf_list, emax_fem, 's-', color=C_FEM, label='FEM')
ax.set_xlabel(r'$v_f$ (%)'); ax.set_ylabel(r'$e_\mathrm{max}$ (%)')
ax.set_title(r'(c) Maximum error vs FFT'); ax.legend()
save(fig, 'fig_series2_vf')

# ============================================================
# SERIES 3 -- Aspect ratio
# ============================================================
ar_list = [1, 2, 5, 10, 20]
Kyy_fft3 = [0.60714, 0.60969, 0.61299, 0.61362, 0.62433]
Kxx_fft3 = [0.60590, 0.60414, 0.60336, 0.60160, 0.61448]
Kyy_lr3  = [0.60952, 0.61154, 0.61358, 0.61408, 0.62433]
Kxx_lr3  = [0.60910, 0.60673, 0.60660, 0.60638, 0.61376]
Kyy_fem3 = [0.60815, 0.61020, 0.61239, 0.61363, 0.62380]
Kxx_fem3 = [0.60750, 0.60557, 0.60490, 0.60507, 0.61719]
aniso_fft = [y/x for x,y in zip(Kxx_fft3, Kyy_fft3)]
aniso_lr  = [y/x for x,y in zip(Kxx_lr3,  Kyy_lr3)]
aniso_fem = [y/x for x,y in zip(Kxx_fem3, Kyy_fem3)]
emax_lr3  = [0.54, 0.57, 0.59, 0.90, 0.19]
emax_fem3 = [0.26, 0.37, 0.29, 0.58, 0.48]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.semilogx(ar_list, Kyy_fft3, 'o-', color=C_FFT, label=r'$K_{yy}$ FFT')
ax.semilogx(ar_list, Kxx_fft3, 's--', color=C_FFT, alpha=0.6, label=r'$K_{xx}$ FFT')
ax.semilogx(ar_list, Kyy_lr3,  'o-', color=C_LR,  label=r'$K_{yy}$ LS-RJF')
ax.semilogx(ar_list, Kxx_lr3,  's--', color=C_LR,  alpha=0.6, label=r'$K_{xx}$ LS-RJF')
ax.semilogx(ar_list, Kyy_fem3, 'o-', color=C_FEM, label=r'$K_{yy}$ FEM')
ax.semilogx(ar_list, Kxx_fem3, 's--', color=C_FEM, alpha=0.6, label=r'$K_{xx}$ FEM')
ax.set_xlabel(r'Aspect ratio $\alpha$'); ax.set_ylabel(r'$K$ [W/(m$\cdot$K)]')
ax.set_title('(a) Diagonal components'); ax.legend(fontsize=8, ncol=2)

ax = axes[1]
ax.semilogx(ar_list, aniso_fft, 'o-', color=C_FFT, label='FFT')
ax.semilogx(ar_list, aniso_lr,  'o-', color=C_LR,  label='LS-RJF')
ax.semilogx(ar_list, aniso_fem, 'o-', color=C_FEM, label='FEM')
ax.axhline(1.0, color='k', lw=1, ls='--', label='Isotropic')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$K_{yy}/K_{xx}$')
ax.set_title(r'(b) Anisotropy ratio'); ax.legend()

ax = axes[2]
ax.semilogx(ar_list, emax_lr3,  'o-', color=C_LR,  label='LS-RJF')
ax.semilogx(ar_list, emax_fem3, 's-', color=C_FEM, label='FEM')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$e_\mathrm{max}$ (%)')
ax.set_title(r'(c) Maximum error vs FFT'); ax.legend()
save(fig, 'fig_series3_alpha')

# ============================================================
# SERIES 4 -- Orientation
# ============================================================
orients = ['O-$x$', 'O-$y$', 'O-$z$', 'O-rand']
# K parallel to fiber axis
K_par_fft = [0.61089, 0.61299, 0.61275, None]
K_par_lr  = [0.61217, 0.61358, 0.61346, None]
K_par_fem = [0.61251, 0.61239, 0.61309, None]
# K transverse
K_tr_fft  = [(0.60099+0.60059)/2, (0.60336+0.60321)/2, (0.60205+0.60382)/2, (0.60665+0.60547+0.60764)/3]
K_tr_lr   = [(0.60567+0.60496)/2, (0.60660+0.60677)/2, (0.60435+0.60789)/2, (0.61243+0.60997+0.61159)/3]
K_tr_fem  = [(0.60490+0.60525)/2, (0.60490+0.60498)/2, (0.60557+0.60580)/2, (0.60792+0.60723+0.60871)/3]
emax_lr4  = [0.78, 0.59, 0.67, 0.95]
emax_fem4 = [0.78, 0.29, 0.58, 0.29]

x = np.arange(4)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
K_par_fft_v = [0.61089, 0.61299, 0.61275, np.nan]
K_par_lr_v  = [0.61217, 0.61358, 0.61346, np.nan]
K_par_fem_v = [0.61251, 0.61239, 0.61309, np.nan]
ax.bar(x-W, K_par_fft_v, W, color=C_FFT, label='FFT', alpha=0.8)
ax.bar(x,   K_par_lr_v,  W, color=C_LR,  label='LS-RJF', alpha=0.8)
ax.bar(x+W, K_par_fem_v, W, color=C_FEM, label='FEM', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(orients)
ax.set_ylabel(r'$K_\parallel$ [W/(m$\cdot$K)]')
ax.set_title('(a) Parallel component'); ax.legend()

ax = axes[1]
ax.bar(x-W, K_tr_fft, W, color=C_FFT, label='FFT', alpha=0.8)
ax.bar(x,   K_tr_lr,  W, color=C_LR,  label='LS-RJF', alpha=0.8)
ax.bar(x+W, K_tr_fem, W, color=C_FEM, label='FEM', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(orients)
ax.set_ylabel(r'$\bar K_\perp$ [W/(m$\cdot$K)]')
ax.set_title('(b) Mean transverse component'); ax.legend()

ax = axes[2]
ax.bar(x-W/2, emax_lr4,  W, color=C_LR,  label='LS-RJF', alpha=0.8)
ax.bar(x+W/2, emax_fem4, W, color=C_FEM, label='FEM', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(orients)
ax.set_ylabel(r'$e_\mathrm{max}$ (%)'); ax.set_title(r'(c) Max error vs FFT'); ax.legend()
save(fig, 'fig_series4_orientation')

# ============================================================
# SERIES 5 -- Resolution
# ============================================================
N_list   = [10, 15, 20, 30]
Ng_list  = [11, 16, 21, 31]
Kxx_fft5 = [0.60844, 0.60861, 0.60355, 0.60336]
Kyy_fft5 = [0.61624, 0.61772, 0.61367, 0.61299]
Kzz_fft5 = [0.60696, 0.60848, 0.60418, 0.60321]
Kxx_lr5  = [0.60844, 0.61270, 0.60618, 0.60660]
Kyy_lr5  = [0.61598, 0.61851, 0.61361, 0.61358]
Kzz_lr5  = [0.60567, 0.61171, 0.60591, 0.60677]
Kxx_fem5 = [0.60273, 0.60105, 0.60451, 0.60490]
Kyy_fem5 = [0.60774, 0.60731, 0.61132, 0.61239]
emax_lr5  = [0.21, 0.67, 0.44, 0.59]
emax_fem5 = [1.38, 1.69, 0.38, 0.29]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.plot(Ng_list, Kyy_fft5, 'o-', color=C_FFT, label=r'$K_{yy}$ FFT')
ax.plot(Ng_list, Kxx_fft5, 's--', color=C_FFT, alpha=0.6, label=r'$K_{xx}$ FFT')
ax.plot(Ng_list, Kyy_lr5,  'o-', color=C_LR,  label=r'$K_{yy}$ LS-RJF')
ax.plot(Ng_list, Kxx_lr5,  's--', color=C_LR,  alpha=0.6, label=r'$K_{xx}$ LS-RJF')
ax.plot(N_list,  Kyy_fem5, 'o-', color=C_FEM, label=r'$K_{yy}$ FEM')
ax.plot(N_list,  Kxx_fem5, 's--', color=C_FEM, alpha=0.6, label=r'$K_{xx}$ FEM')
ax.set_xlabel(r'Grid size $N_g$ (FFT/LR) or $N$ (FEM)')
ax.set_ylabel(r'$K$ [W/(m$\cdot$K)]'); ax.set_title('(a) Diagonal components'); ax.legend(fontsize=8)

ax = axes[1]
Kxy_fem5 = [0.00415, 0.00192, 0.00114, 0.00055]
ax.plot(N_list, Kxy_fem5, 's-', color=C_FEM, label='|Kxy| FEM')
ax.set_xlabel('$N$'); ax.set_ylabel(r'$|K_{xy}|$ [W/(m$\cdot$K)]')
ax.set_title('(b) Off-diagonal FEM convergence'); ax.legend()

ax = axes[2]
ax.plot(Ng_list, emax_lr5,  'o-', color=C_LR,  label='LS-RJF')
ax.plot(N_list,  emax_fem5, 's-', color=C_FEM, label='FEM')
ax.set_xlabel(r'$N_g$ (LS-RJF) or $N$ (FEM)')
ax.set_ylabel(r'$e_\mathrm{max}$ (%)'); ax.set_title(r'(c) Max error vs FFT'); ax.legend()
save(fig, 'fig_series5_resolution')

# ============================================================
# SERIES 6 -- kf sweep
# ============================================================
kf_vals = [0.05, 0.10, 0.20, 0.29, 0.50, 1.00]
labels6 = ['coir\n(0.05)', 'sisal\n(0.10)', 'poly.\n(0.20)',
           'base\n(0.29)', 'PE\n(0.50)', 'glass\n(1.00)']
Kyy_fft6 = [0.58446, 0.59079, 0.60275, 0.61299, 0.63558, 0.68471]
Kxx_fft6 = [0.54475, 0.56081, 0.58577, 0.60336, 0.63424, 0.67984]
Kyy_lr6  = [0.58809, 0.59316, 0.60412, 0.61358, 0.63557, 0.68535]
Kxx_lr6  = [0.56083, 0.57087, 0.59147, 0.60660, 0.63449, 0.68220]
Kyy_fem6 = [0.58491, 0.59091, 0.60241, 0.61239, 0.63471, 0.68413]
Kxx_fem6 = [0.55567, 0.56851, 0.58944, 0.60490, 0.63361, 0.68025]
aniso_fft6 = [y-x for x,y in zip(Kxx_fft6, Kyy_fft6)]
aniso_lr6  = [y-x for x,y in zip(Kxx_lr6,  Kyy_lr6)]
emax_lr6  = [3.45, 2.27, 1.16, 0.59, 0.04, 0.35]
emax_fem6 = [2.35, 1.56, 0.70, 0.29, 0.14, 0.09]

x6 = np.arange(len(kf_vals))
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ax = axes[0]
ax.plot(x6, Kyy_fft6, 'o-', color=C_FFT, label=r'$K_{yy}$ FFT')
ax.plot(x6, Kxx_fft6, 's--', color=C_FFT, alpha=0.6, label=r'$K_{xx}$ FFT')
ax.plot(x6, Kyy_lr6,  'o-', color=C_LR,  label=r'$K_{yy}$ LS-RJF')
ax.plot(x6, Kxx_lr6,  's--', color=C_LR,  alpha=0.6, label=r'$K_{xx}$ LS-RJF')
ax.plot(x6, Kyy_fem6, 'o-', color=C_FEM, label=r'$K_{yy}$ FEM')
ax.plot(x6, Kxx_fem6, 's--', color=C_FEM, alpha=0.6, label=r'$K_{xx}$ FEM')
ax.axhline(0.651, color='k', ls=':', lw=1.2, label=r'$k_m$')
ax.set_xticks(x6); ax.set_xticklabels(labels6)
ax.set_ylabel(r'$K$ [W/(m$\cdot$K)]'); ax.set_title('(a) Diagonal components'); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(x6, aniso_fft6, 'o-', color=C_FFT, label='FFT')
ax.plot(x6, aniso_lr6,  'o-', color=C_LR,  label='LS-RJF')
ax.axhline(0, color='k', ls='--', lw=1)
ax.set_xticks(x6); ax.set_xticklabels(labels6)
ax.set_ylabel(r'$K_{yy}-K_{xx}$ [W/(m$\cdot$K)]')
ax.set_title(r'(b) Anisotropy $K_{yy}-K_{xx}$'); ax.legend()

ax = axes[2]
ax.plot(x6, emax_lr6,  'o-', color=C_LR,  label='LS-RJF')
ax.plot(x6, emax_fem6, 's-', color=C_FEM, label='FEM')
ax.set_xticks(x6); ax.set_xticklabels(labels6)
ax.set_ylabel(r'$e_\mathrm{max}$ (%)'); ax.set_title(r'(c) Max error vs FFT'); ax.legend()
save(fig, 'fig_series6_kfiber')

# ============================================================
# SERIES 7 -- k0 sweep
# ============================================================
k0_labels  = [r'$k_f$\n(div)', r'$k_f{+}0.1\Delta k$', r'$(k_m{+}k_f)/2$', r'$k_m{-}0.1\Delta k$', r'$k_m$']
k0_vals    = [0.290, 0.326, 0.471, 0.615, 0.651]
Kyy_lr7    = [0.61410, 0.61395, 0.61358, 0.61380, 0.61345]
Kxx_lr7    = [0.61168, 0.60733, 0.60660, 0.60642, 0.60618]
# FFT reference (constant for k0 != kf_case)
Kyy_fft7   = [np.nan,  0.61299, 0.61299, 0.61299, 0.61299]
Kxx_fft7   = [np.nan,  0.60336, 0.60336, 0.60336, 0.60336]
emax_lr7   = [np.nan,  0.75,    0.59,    0.57,    0.54]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
x7 = np.arange(5)
ax = axes[0]
ax.plot(x7, Kyy_lr7,  'o-', color=C_LR,  label=r'$K_{yy}$ LS-RJF')
ax.plot(x7, Kxx_lr7,  's-', color=C_LR,  alpha=0.6, label=r'$K_{xx}$ LS-RJF', ls='--')
ax.plot(x7, Kyy_fft7, 'o--', color=C_FFT, label=r'$K_{yy}$ FFT')
ax.plot(x7, Kxx_fft7, 's--', color=C_FFT, alpha=0.6, label=r'$K_{xx}$ FFT')
ax.axvline(0, color='red', ls=':', lw=1.5, label='FFT diverges')
ax.set_xticks(x7)
ax.set_xticklabels([r'$k_f$', r'$k_f{+}0.1\Delta$', r'base', r'$k_m{-}0.1\Delta$', r'$k_m$'], fontsize=9)
ax.set_ylabel(r'$K$ [W/(m$\cdot$K)]'); ax.set_title(r'(a) $K_{xx}$, $K_{yy}$ vs $k_0$'); ax.legend(fontsize=8)

ax = axes[1]
dev_yy = [y - 0.61358 for y in Kyy_lr7]
dev_xx = [x - 0.60660 for x in Kxx_lr7]
ax.bar(x7-W/2, [d*1000 for d in dev_yy], W, color=C_LR, label=r'$\Delta K_{yy}$', alpha=0.8)
ax.bar(x7+W/2, [d*1000 for d in dev_xx], W, color='orange', label=r'$\Delta K_{xx}$', alpha=0.8)
ax.set_xticks(x7)
ax.set_xticklabels([r'$k_f$', r'$k_f{+}0.1\Delta$', r'base', r'$k_m{-}0.1\Delta$', r'$k_m$'], fontsize=9)
ax.set_ylabel(r'Deviation from baseline [$\times10^{-3}$ W/(m$\cdot$K)]')
ax.set_title('(b) Deviation from baseline $k_0$'); ax.legend()

ax = axes[2]
x7v = [1,2,3,4]; e7v = [0.75, 0.59, 0.57, 0.54]
ax.plot(x7v, e7v, 'o-', color=C_LR, label='LS-RJF')
ax.set_xticks(x7v)
ax.set_xticklabels([r'$k_f{+}0.1\Delta$', r'base', r'$k_m{-}0.1\Delta$', r'$k_m$'], fontsize=9)
ax.set_ylabel(r'$e_\mathrm{max}$ (%)'); ax.set_title(r'(c) Max error vs FFT (FFT diverges at $k_0=k_f$)'); ax.legend()
save(fig, 'fig_series7_k0')

print('\nAll figures saved.')

# ============================================================
# SERIES 9 -- Number of inclusions
# ============================================================
ninc_list = [10, 20, 30, 40, 50]

Kxx_fft9 = [0.6036, 0.6034, 0.6044, 0.6037, 0.6034]
Kyy_fft9 = [0.6132, 0.6130, 0.6139, 0.6136, 0.6134]
Kzz_fft9 = [0.6033, 0.6032, 0.6042, 0.6037, 0.6039]

Kxx_lr9  = [0.6061, 0.6066, 0.6078, 0.6081, 0.6090]
Kyy_lr9  = [0.6136, 0.6136, 0.6148, 0.6141, 0.6149]
Kzz_lr9  = [0.6051, 0.6068, 0.6081, 0.6079, 0.6096]

Kxx_fem9 = [0.6048, 0.6049, 0.6056, 0.6058, 0.6063]
Kyy_fem9 = [0.6130, 0.6124, 0.6129, 0.6129, 0.6132]
Kzz_fem9 = [0.6054, 0.6050, 0.6058, 0.6059, 0.6065]

emax_lr9  = [0.41, 0.59, 0.65, 0.74, 0.94]
emax_fem9 = [0.35, 0.29, 0.26, 0.37, 0.48]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

ax = axes[0]
ax.plot(ninc_list, Kyy_fft9, 'o-', color=C_FFT, label=r'$K_{yy}$ FFT')
ax.plot(ninc_list, Kxx_fft9, 's--', color=C_FFT, alpha=0.6, label=r'$K_{xx}$ FFT')
ax.plot(ninc_list, Kyy_lr9,  'o-', color=C_LR,  label=r'$K_{yy}$ LS-RJF')
ax.plot(ninc_list, Kxx_lr9,  's--', color=C_LR,  alpha=0.6, label=r'$K_{xx}$ LS-RJF')
ax.plot(ninc_list, Kyy_fem9, 'o-', color=C_FEM, label=r'$K_{yy}$ FEM')
ax.plot(ninc_list, Kxx_fem9, 's--', color=C_FEM, alpha=0.6, label=r'$K_{xx}$ FEM')
ax.set_xlabel(r'Number of inclusions $n$')
ax.set_ylabel(r'$K$ [W/(m$\cdot$K)]')
ax.set_title('(a) Diagonal components')
ax.legend(fontsize=8, ncol=2)

ax = axes[1]
ax.plot(ninc_list, Kzz_fft9, 'o-', color=C_FFT, label=r'$K_{zz}$ FFT')
ax.plot(ninc_list, Kzz_lr9,  'o-', color=C_LR,  label=r'$K_{zz}$ LS-RJF')
ax.plot(ninc_list, Kzz_fem9, 'o-', color=C_FEM, label=r'$K_{zz}$ FEM')
ax.set_xlabel(r'Number of inclusions $n$')
ax.set_ylabel(r'$K_{zz}$ [W/(m$\cdot$K)]')
ax.set_title(r'(b) $K_{zz}$ component')
ax.legend()

ax = axes[2]
ax.plot(ninc_list, emax_lr9,  'o-', color=C_LR,  label='LS-RJF')
ax.plot(ninc_list, emax_fem9, 's-', color=C_FEM, label='FEM')
ax.axvline(20, color='k', ls=':', lw=1.5, label='Baseline $n=20$')
ax.set_xlabel(r'Number of inclusions $n$')
ax.set_ylabel(r'$e_\mathrm{max}$ (%)')
ax.set_title(r'(c) Maximum error vs FFT')
ax.legend()

save(fig, 'fig_series9_ninc')
print('\nSeries 9 figure saved.')
