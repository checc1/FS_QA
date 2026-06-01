import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import curve_fit



def load_dims_for_tau(tau, beta=0.7, F=16, concat=True):
    datas = []
    for ci in range(10):
        #pkl = f"/Users/francescoaldoventurelli/qml/projet_one/dictonaries/tau{tau}/class{ci}/probs/fidelity_Dict_class{ci}_beta_{beta}model{F}.pkl"
        pkl = os.path.join(os.getcwd(), "Dictonaries", f"tau{tau}", f"class{ci}", "fidelity", f"fidelity_Dict_class{ci}_beta_0.7model{F}.pkl")
        with open(pkl, "rb") as f:
            datas.append(pickle.load(f))
    if concat:
        concatenation = np.concatenate([d["dim"] for d in datas], axis=0)
        return concatenation
    else:
        return np.array(datas)

def create_p_tau(tau):
    dims = load_dims_for_tau(tau)
    #dims = np.concatenate([d["dim"] for d in datas], axis=0)

    #p = np.load(f"/Users/francescoaldoventurelli/qml/projet_one/fids_tau-{tau}.npy")

    unique_dims = np.unique(dims)

    infidelity_dim = []

    std_infidelity_max = []
    std_infidelity_min = []


    for d_val in unique_dims:
        mask = (dims == d_val)
        q25, q50, q75 = np.percentile(p[mask], [25, 50, 75])

        infidelity_dim.append(q50)

        std_infidelity_min.append(q50 - q25)
        std_infidelity_max.append(q75 - q50)

    return (infidelity_dim, std_infidelity_max, std_infidelity_min)


### If you want to run the plot_all() attiva questo
"""(fid10, fidmax10, fidmin10) = create_p_tau(10)
(fid30, fidmax30, fidmin30) = create_p_tau(30)
(fid50, fidmax50, fidmin50) = create_p_tau(50)
(fid70, fidmax70, fidmin70) = create_p_tau(70)
(fid100, fidmax100, fidmin100) = create_p_tau(100)"""


def plot_all():
    f_list = {"10": fid10, "30": fid30, "50": fid50, "70": fid70, "100": fid100}
    f_max_list = [fidmax10, fidmax30, fidmax50, fidmax70, fidmax100]
    f_min_list = [fidmin10, fidmin30, fidmin50, fidmin70, fidmin100]
    yerr_list = [np.vstack([fmax, fmin]) for fmax, fmin in zip(f_max_list, f_min_list)]
    colors = {"10": "#440154FF", "30": "#3B528BFF", "50": "#21918CFF", "70": "#7AD151FF", "100": "#FDE725FF"}

    unique_dims = np.array([3, 5, 6, 7 ,8])
    for f_key,errf in zip(f_list.keys(), yerr_list):
        plt.errorbar(
            unique_dims,
            f_list[f_key],
            yerr=errf,
            fmt="o",
            capsize=5,
            capthick=1,
            elinewidth=1,
            label=fr"$\tau={f_key}$",
            color=colors[f_key],
            #markeredgecolor="k",
            markeredgewidth=0.75,
            markersize=6,
            linestyle="none",
        )
    plt.xlabel(r"$d$", fontsize=16)
    plt.ylabel(r"$\langle F \rangle$", fontsize=16)
    plt.yscale("log")
    plt.legend(fontsize=12, frameon=True, fancybox=True, edgecolor="k")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("/Users/francescoaldoventurelli/Downloads/fidelity_per_dim.pdf")
    plt.show()

#plot_all()

def p_per_class(tau):
    dims = load_dims_for_tau(tau, concat=False)
    #dims = np.concatenate([d["dim"] for d in datas], axis=0)

    p = np.load(f"/Users/francescoaldoventurelli/qml/projet_one/fids_tau-{tau}.npy")

    unique_dims = np.unique(dims)

    infidelity_dim = []

    std_infidelity_max = []
    std_infidelity_min = []


    for d_val in unique_dims:
        mask = (dims == d_val)
        q25, q50, q75 = np.percentile(p[mask], [25, 50, 75])

        infidelity_dim.append(q50)

        std_infidelity_min.append(q50 - q25)
        std_infidelity_max.append(q75 - q50)

    return (infidelity_dim, std_infidelity_max, std_infidelity_min)


def summarize_per_class(dims, stat="median"):
    center, err_low, err_high = [], [], []
    for j in range(10):
        vals = np.asarray(dims[j]["probs"])[:, -1]

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if stat == "mean":
            vc = float(np.mean(vals))
        else:
            vc = float(np.median(vals))

        center.append(vc)
        err_low.append(vc - vmin)
        err_high.append(vmax - vc)

    return np.array(center), np.array(err_low), np.array(err_high)


def plot_fidelity_by_tau(tauList, stat="median"):

    (fid10, fidmax10, fidmin10) = create_p_tau(10)
    (fid30, fidmax30, fidmin30) = create_p_tau(30)
    #(fid50, fidmax50, fidmin50) = create_p_tau(50)
    #(fid70, fidmax70, fidmin70) = create_p_tau(70)
    #(fid100, fidmax100, fidmin100) = create_p_tau(100)

    f_list = {"10": fid10, "30": fid30}
    f_max_list = [fidmax10, fidmax30]
    f_min_list = [fidmin10, fidmin30]
    yerr_list = [np.vstack([fmax, fmin]) for fmax, fmin in zip(f_max_list, f_min_list)]
    #colors = {"10": "#003f5c", "30": "#58508d", "50": "#bc5090", "70": "#ff6361", "100": "#ffa600"}
    colors = {"10": "#003f5c", "30": "#58508d"}

    unique_dims = np.array([7,8,9,11])
    p10 = f_list["10"]

    def func(x, a, b, c, d):
        return a * np.exp(-c * (x - b)) + d


    popt, pcov = curve_fit(func, unique_dims, p10)
    xfit = np.linspace(unique_dims.min(), unique_dims.max(), 200)
    yfit = func(xfit, *popt)


    #markers = {10: "o", 30:"s", 50:"^", 70:"v", 100:"D"}
    markers = {10: "o", 30:"s"}
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    for f_key, errf in zip(f_list.keys(), yerr_list):
        axs[0].errorbar(
            unique_dims,
            f_list[f_key],
            yerr=errf,
            fmt=markers[int(f_key)],
            capsize=4,
            capthick=1,
            elinewidth=0.7,
            linewidth=0.9,
            label=fr"$\tau={f_key}$",
            color=colors[f_key],
            # markeredgecolor="k",
            markeredgewidth=0.75,
            markersize=7,
            linestyle="none",
        )
    axs[0].plot(xfit, yfit, linewidth=1.5, color="red", linestyle="--",)
    per_tau_centers = {}
    for k, tau in enumerate(tauList):
        dims = load_dims_for_tau(tau, concat=False)
        center, err_low, err_high = summarize_per_class(dims, stat=stat)
        per_tau_centers[tau] = center

        yerr = np.vstack([err_low, err_high])

        axs[1].errorbar(
            range(10),
            center,
            yerr=yerr,
            fmt=markers[tau],
            linestyle="none",
            capsize=4,
            elinewidth=0.7,
            linewidth=0.9,
            color=colors[str(tau)],
            label=rf"$\tau={tau}$",
            markersize=7,
        )

    axs[0].set_xlabel(r"$d$", fontsize=15)
    axs[1].set_xlabel(r"$\text{Class}$", fontsize=15)
    axs[0].set_ylabel(r"$\overline{F}$", fontsize=16)
    #axs[1].set_ylabel(r"$\overline{F}$", fontsize=16)
    #dims = [r"$2^3$", r"$2^5$", r"$2^6$", r"$2^7$", r"$2^8$"]
    axs[0].set_xticks(unique_dims, unique_dims, fontsize=15)
    axs[0].set_yticks(unique_dims, f_list, fontsize=15)
    axs[1].set_yticks([])
    #axs[1].set_xticks(range(10), ["Airplane", "Bird", "Car", "Cat", "Deer", "Dog", "Horse", "Monkey", "Ship", "Truck"], fontsize=13, rotation=60)
    axs[1].set_xticks(range(10), list(range(10)), fontsize=15)
    #axs[1].set_xlim(-0.5, 9.5)

    for ax in axs:
        ax.set_yscale("log")
        ax.set_ylim(2e-2, 1.2)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="major", linestyle=":", linewidth=0.7)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.6)
    axs[0].legend(fontsize=12, frameon=True, fancybox=True, edgecolor="k")

    plt.tight_layout()
    #plt.savefig("/Users/francescoaldoventurelli/Downloads/fidelity_per_class_AND_per_dim.pdf")
    plt.show()

    return per_tau_centers


def load_fidelity_dict(tau, class_id, beta=0.7, model=16, base_dir="Dictonaries"):
    filepath = os.path.join(
        base_dir,
        f"tau{tau}",
        f"class{class_id}",
        "fidelity",
        f"fidelity_Dict_class{class_id}_beta_{beta}model{model}.pkl"
    )

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data


def load_all_classes_for_tau(tau, num_classes=10):
    data = {}
    for c in range(num_classes):
        data[c] = load_fidelity_dict(tau, c)
    return data

def load_all_data(tauList, num_classes=10):
    all_data = {}
    for tau in tauList:
        all_data[tau] = load_all_classes_for_tau(tau, num_classes)
    return all_data



"""all_fidelity = load_all_data([10, 30])
data = load_fidelity_dict(10, 1)
print(data.keys())
print(data["dim"])"""

from collections import defaultdict

def aggregate_dims(data):

    dim_groups = defaultdict(list)

    dims = data["dim"]
    fidelities = data["probs"]

    for d, f in zip(dims, fidelities):
        dim_groups[d].append(f)

    return dim_groups

def plot_tau_lines_by_class(tauList, n_classes=10):

    markers = {10:"o", 30:"s", 50:"^", 70:"v", 100:"x"}
    colors = {10:"#003f5c", 30:"#58508d", 50:"#bc5090", 70:"#ff6361", 100: "#ffa600"}

    classes = np.arange(n_classes)

    plt.figure(figsize=(5.5,4))

    for tau in tauList:

        class_values = []
        std_infidelity_min = []
        std_infidelity_max = []

        for c in range(n_classes):

            data = load_fidelity_dict(tau, c)
            values = np.concatenate([np.ravel(v) for v in data.values()])

            q25, q50, q75 = np.percentile(values, [25, 50, 75])

            std_infidelity_min.append(q50 - q25)
            std_infidelity_max.append(q75 - q50)

            dim_groups = aggregate_dims(data)
            dim_medians = [np.median(v) for v in dim_groups.values()]
            class_values.append(np.median(dim_medians))

        yerr = [std_infidelity_min, std_infidelity_max]

        plt.errorbar(
            classes,
            class_values,
            yerr=yerr,
            marker=markers.get(tau,"o"),
            color=colors.get(tau),
            linewidth=1.5,
            markersize=7,
            label=rf"$\tau={tau}$",
            linestyle="none",
            capsize=4
        )

    plt.xlabel("Class", fontsize=18)
    plt.ylabel(r"$\overline{F}$", fontsize=20)
    plt.yscale("log")
    plt.ylim(0.09, 1.1)
    plt.xticks(classes, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(fancybox=True, frameon=True, fontsize=14, edgecolor="k")
    plt.tight_layout()
    plt.show()

#plot_tau_lines_by_class([10, 30, 50, 70, 100], n_classes=10)

def aggregate_dims(data):
    """
    Group fidelities by dimension.
    Expects:
        data["dim"]   -> iterable of dimensions
        data["probs"] -> iterable of fidelities
    """
    dim_groups = defaultdict(list)

    dims = data["dim"]
    fidelities = data["fidelity"]

    for d, f in zip(dims, fidelities):
        dim_groups[d].append(f)

    return dim_groups


def exp_func(x, a, b, c, d):
    return a * np.exp(-c * (x - b)) + d


def plot_tau_subplots(tauList, n_classes=10):
    markers = {10: "o", 30: "s", 50: "^", 70: "v", 100: "x"}
    colors = {
        10: "#003f5c",
        30: "#58508d",
        50: "#bc5090",
        70: "#ff6361",
        100: "#ffa600"
    }

    classes = np.arange(n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.))
    ax1, ax2 = axes


    for tau in tauList:
        class_values = []
        qerr_low = []
        qerr_high = []

        for c in range(n_classes):
            data = load_fidelity_dict(tau, c)

            values = np.ravel(np.asarray(data["fidelity"]))

            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            qerr_low.append(q50 - q25)
            qerr_high.append(q75 - q50)

            dim_groups = aggregate_dims(data)
            dim_medians = [np.median(v) for v in dim_groups.values()]
            class_values.append(np.median(dim_medians))

        ax2.errorbar(
            classes,
            class_values,
            yerr=[qerr_low, qerr_high],
            marker=markers.get(tau, "o"),
            color=colors.get(tau, None),
            linewidth=1.5,
            markersize=8,
            label=rf"$\tau={tau}$",
            linestyle="none",
            capsize=5,
            #label=fr"$\tau={tau}$"
        )

    ax2.set_xlabel("Class", fontsize=20)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.set_yscale("log")
    ax2.set_ylim(0.09, 1.1)
    ax2.set_xticks(classes)
    ax2.tick_params(axis="both", labelsize=16)
    ax2.grid(True, which="both", linestyle=":", alpha=0.8)


    for tau in tauList:
        all_dim_groups = defaultdict(list)


        for c in range(n_classes):
            data = load_fidelity_dict(tau, c)
            dim_groups = aggregate_dims(data)

            for d, vals in dim_groups.items():
                all_dim_groups[d].extend(np.ravel(vals))

        unique_dims = np.array(sorted(all_dim_groups.keys()))

        medians10 = np.array([np.median(all_dim_groups[d]) for d in unique_dims])
        medians = np.array([np.median(all_dim_groups[d]) for d in unique_dims])
        q25 = np.array([np.percentile(all_dim_groups[d], 25) for d in unique_dims])
        q75 = np.array([np.percentile(all_dim_groups[d], 75) for d in unique_dims])

        yerr = [medians - q25, q75 - medians]

        ax1.errorbar(
            unique_dims,
            medians,
            yerr=yerr,
            marker=markers.get(tau, "o"),
            color=colors.get(tau, None),
            linewidth=1.5,
            markersize=8,
            linestyle="none",
            capsize=5,
            label=rf"$\tau={tau}$"
        )
        ax1.set_ylim(0.09, 1.1)
        ax1.set_xticks(unique_dims)
    ax1.tick_params(axis="y", labelleft=True)
    ax1.set_xticklabels(unique_dims, fontsize=16)
    ax1.tick_params(axis="y", labelsize=16)

    #ax2.yaxis.set_visible(False)
    ax1.set_xlabel(r"$d$", fontsize=20)
    #ax2.set_ylabel(r"$\overline{F}$", fontsize=20)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.8)
    ax1.tick_params(length=8, axis="x")
    ax1.tick_params(length=8, axis="y")
    ax2.tick_params(length=8, axis="y")
    ax2.tick_params(length=8, axis="x")
    ax1.set_ylabel(r"$\overline{F}$", fontsize=20)
    ax1.legend(fancybox=True, frameon=True, fontsize=12, edgecolor="k")

    #ax2.legend(fancybox=True, frameon=True, fontsize=11, edgecolor="k")

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "figures", "fidelity_dim_updated.pdf"))
    plt.show()

plot_tau_subplots([10, 30, 50, 70, 100], n_classes=10)