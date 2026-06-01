import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import ScalarFormatter, FixedLocator


nfeature_list=[16,24,32]
path = "/Users/francescoaldoventurelli/Downloads"
dir_names=[os.path.join(path, f"model_{n}_features_3") for n in nfeature_list]


gaps_list=[]

for dir_name in dir_names:
    with open(f"{dir_name}/gaps.pkl", "rb") as f:
        gaps = pickle.load(f)
    gaps_list.append(np.asarray(gaps))
mintime_list=[]
for dir_name in dir_names:
    with open(f"{dir_name}/mintimes.pkl", "rb") as f:
        mintime = pickle.load(f)
    mintime_list.append(np.asarray(mintime))
gap_vs_class_list=[]
for dir_name in dir_names:
    with open(f"{dir_name}/gap_vs_class.pkl", "rb") as f:
        gap_vs_class = pickle.load(f)
    gap_vs_class_list.append(gap_vs_class)
gap_vs_dimension_list=[]
for dir_name in dir_names:
    with open(f"{dir_name}/gap_vs_dimension.pkl", "rb") as f:
        gap_vs_dimension = pickle.load(f)
    gap_vs_dimension_list.append(gap_vs_dimension)


hardest_gap_per_nf = {}
for nf, gap_vs_dimension in zip(nfeature_list, gap_vs_dimension_list):
    hardest_gap_per_nf[nf] = {}
    for dim, gaps in gap_vs_dimension.items():
        hardest_gap_per_nf[nf][dim] = float(np.min(gaps))

hardest_gap_global = {}
for d in set().union(*[g.keys() for g in gap_vs_dimension_list]):
    hardest_gap_global[d] = min(
        np.min(g[d]) for g in gap_vs_dimension_list if d in g
    )


hardest_gap_per_nf = {}

for nf, gap_vs_dimension in zip(nfeature_list, gap_vs_dimension_list):
    hardest_gap_per_nf[nf] = {}

    for dim, gaps in gap_vs_dimension.items():
        flat_gaps = np.array(gaps).flatten()
        hardest_gap_per_nf[nf][dim] = float(np.min(flat_gaps))

merged_min = {}

dims = set().union(*[d.keys() for d in hardest_gap_per_nf.values()])

for d in dims:
    best_val = np.inf
    best_nf = None

    for nf in hardest_gap_per_nf:
        if d in hardest_gap_per_nf[nf]:
            val = hardest_gap_per_nf[nf][d]

            if val < best_val:
                best_val = val
                best_nf = nf

    merged_min[d] = (best_val, best_nf)




def plot():
    all_dimensions = []
    plt.figure(figsize=(10, 6))
    all_means = []; all_stds = []

    for i, gap_vs_dimension in enumerate(gap_vs_dimension_list):
        mean_per_dimension = []
        std_per_dimension = []
        dims = np.array(sorted(list(gap_vs_dimension.keys())))
        for dimension, gaps in gap_vs_dimension.items():
            mean_per_dimension.append(np.mean(gaps))
            std_per_dimension.append(np.std(gaps))
            #all_dimensions.extend(dimensions)

        all_means.append(mean_per_dimension)
        all_stds.append(std_per_dimension)
        plt.errorbar(dims, mean_per_dimension, yerr=std_per_dimension, fmt='o', capsize=5,
                     label=f'N_f={nfeature_list[i]}')
        all_dimensions.append(dims)
    all_dimensions = np.concatenate(np.array(all_dimensions))
    all_means = [float(x) for xs in all_means for x in xs]
    all_stds = [x for xs in all_stds for x in xs]
    log_x = np.log(all_dimensions)
    log_y_mean = np.log(all_means)
    slope_mean, intercept_mean, r_value, p_value, std_err = linregress(log_x, log_y_mean)
    fitted_mean = np.exp(intercept_mean) * all_dimensions ** slope_mean

    # Log-log regression for std
    log_y_std = np.log(all_stds)
    slope_std, intercept_std, r_value, p_value, std_err = linregress(log_x, log_y_std)
    fitted_std = np.exp(intercept_std) * all_dimensions ** slope_std
    label_ticks = []
    plt.xticks((label_ticks.append(list(gap_vs_dimension.keys()) for gap_vs_dimension in gap_vs_dimension_list)))
    plt.xlabel('dimension', fontsize=20)
    plt.ylabel(r"$\overline{\Delta_{\text{min}}}$", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(all_dimensions, fitted_std, ':', label=r'$\overline{\Delta}$', fontsize=20)

    plt.legend(fontsize=15)

    plt.grid(True, ls='--', alpha=0.4)
    plt.loglog()
    plt.xticks(np.arange(4, 20, 2), np.arange(4, 20, 2))
    plt.show()



def plot_gap_scaling(gap_vs_dimension_list, nfeature_list, fit_what="median"):

    fig, ax = plt.subplots(figsize=(8, 4.5))

    all_dims = []
    all_y = []
    markers = {range(5, 9): "o", range(7, 12): "s", range(10, 16): "v", range(15, 19): "^"}
    colors = {range(5, 9): "#bfd3e6", range(7, 12): "#8c96c6", range(10, 16): "#810f7c", range(15, 19): "#ED5958"}
    low_errs, upp_errs = [], []
    for (i, gap_vs_dimension), key in zip(enumerate(gap_vs_dimension_list), markers.keys()):
        dims = np.array(sorted(gap_vs_dimension.keys()), dtype=int)

        means = np.array([np.median(gap_vs_dimension[d]) for d in dims], dtype=float)
        stdsMax  = np.array([np.percentile(gap_vs_dimension[d], 75)  for d in dims], dtype=float)
        stdsMin = np.array([np.percentile(gap_vs_dimension[d], 25) for d in dims], dtype=float)
        first, third = [m - minS for m, minS in zip(means, stdsMin)], [maxS - m for m, maxS in zip(means, stdsMax)]
        stds = np.vstack([first, third])
        lower = means-stdsMax
        upper = stdsMin-means

        ax.errorbar(
            dims, means, yerr=stds,
            fmt=markers[key], capsize=5, linewidth=0.9, markersize=8, markeredgecolor="gray",
            label=fr"$N_f={nfeature_list[i]}$", color=colors[key]
        )

        low_errs.extend(lower)
        upp_errs.extend(upper)

        y = means if fit_what == "median" else stds

        all_dims.append(dims)
        all_y.append(y)

    all_dims = np.concatenate(all_dims)
    all_y = np.concatenate(all_y)

    mask = (all_dims > 0) & (all_y > 0)
    x = all_dims[mask]
    y = all_y[mask]

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.exp(intercept) * x_fit ** slope

    ax.plot(
        x_fit, y_fit, "--", color="seagreen", linewidth=2,
        label=fr"$\propto d^{{{slope:.2f}}}$"
    )


    ax.set_xscale("log")
    ax.set_yscale("log")

    all_tick_dims = sorted(set(np.concatenate([np.array(sorted(g.keys())) for g in gap_vs_dimension_list])))

    ax.xaxis.set_major_locator(FixedLocator(all_tick_dims))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))

    ax.ticklabel_format(style='plain', axis='x')

    ax.set_xlabel(r"$d$", fontsize=18)
    ax.set_ylabel(r"$\overline{\Delta_{\min}}$", fontsize=18)
    ax.tick_params(axis="x", which="major", labelsize=16, rotation=30)
    ax.tick_params(axis="y", which="major", labelsize=16)

    ax.set_xticks(all_tick_dims)
    ax.set_xticklabels([str(int(t)) for t in all_tick_dims])
    ax.set_ylim(0.06e-4, 10)
    ax.grid(True, which="both", ls="--", alpha=0.5, linewidth=1, zorder=0)
    ax.legend(fontsize=12, frameon=True, fancybox=True, edgecolor="k")
    colors = {
        16: "#FFA500",  # orange
        24: "#F97306",  # darker orange
        32: "#FF4500"  # red-orange (fixed)
    }
    used_labels = set()

    for d, (gap, nf) in merged_min.items():
        label = fr"$\Delta^*_{{N_f={nf}}}$" if nf not in used_labels else None
        used_labels.add(nf)

        ax.scatter(
            d, gap,
            color=colors[nf],
            s=80,
            marker='x',
            label=label
        )


    ax.legend(ncol=2, fontsize=10, frameon=True, fancybox=True, edgecolor="k", loc="lower left")

    plt.tight_layout()
    plt.savefig("/Users/francescoaldoventurelli/Downloads/scalingDelta.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    #return slope, intercept, r_value, p_value, std_err


plot_gap_scaling(gap_vs_dimension_list, nfeature_list)