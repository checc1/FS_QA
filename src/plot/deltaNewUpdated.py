import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

tau = 10
#path = os.path.join("/Users/francescoaldoventurelli/qml/projet_one/quantumStateEvolution")
path = os.path.join("AnnealingWORK", "SavedStates")
c = list(range(10))


def plotDelta(tau: int, excited_state: int):

    Delta = [[] for _ in range(len(c))]

    for ci in range(len(c)):
        spectrum_path = os.path.join(
            path, f"tau{tau}", f"class{ci}", "spectrums"
        )

        for file in os.listdir(spectrum_path):
            energies = np.load(os.path.join(spectrum_path, file))

            e1 = energies[:, excited_state]
            e0 = energies[:, 0]
            delta = float(np.min(e1 - e0))

            Delta[ci].append(delta)

    return Delta

def singlePlot(tau: int):

    excited_state = 1
    delta = plotDelta(tau, excited_state)
    all_delta = np.concatenate(delta)

    bins = np.logspace(
        np.log10(np.min(all_delta)),
        np.log10(np.max(all_delta)),
        50
    )

    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax2 = ax1.twinx()

    counts, bin_edges, _ = ax1.hist(
        all_delta,
        bins=bins,
        alpha=0.5,
        density=False,
        color="#8c96c6",
        edgecolor="indianred"
    )

    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax2.plot(
        bin_centers,
        cdf,
        linewidth=2,
        color="#4d004b",
        label="Cumulative"
    )

    ax1.set_xscale("log")
    ax1.set_xlim(10e-4, 0.25*10e-1)
    ax1.set_xlabel(r"$\Delta_{\text{min}}$", fontsize=26)
    ax1.set_ylabel("Count", fontsize=18    )
    #ax2.set_ylabel(r"$\text{Prob}(\Delta_{\text{min}})$", fontsize=20)
    ax2.set_ylabel(r"CDF", fontsize=18)
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax1.set_ylim(0, 20.5)
    ax2.set_ylim(0, 1.05)
    ax1.tick_params(length=5, axis="x")
    ax1.tick_params(length=5, axis="y")
    ax2.tick_params(length=5, axis="y")
    #ax2.set_yticks(0.1*np.arange(0, 21, 5), [0.00, 0.25, 0.50, 0.75, 1.00])


    ax1.grid(True, which="both", linestyle=":", linewidth=0.7)

    """ax2.legend(
        #[rf"$\tau = {tau}$"],
        fontsize=16,
        edgecolor="black",
        frameon=True,
        fancybox=True,
        #title_fontsize=20
    )"""

    plt.tight_layout()
    #plt.show()
    plt.savefig(
        "DELTA_finalUpdated.pdf", dpi=300, bbox_inches="tight")
    plt.show()

singlePlot(tau)

def multiPlot():
    delta1 = plotDelta(10, 1)
    delta2 = plotDelta(10, 2)
    delta3 = plotDelta(10, 3)
    delta4 = plotDelta(10, 4)
    delta5 = plotDelta(10, 5)
    allD1 = [
            x
            for xs in delta1
            for x in xs
        ]
    allD2 = [
        x
        for xs in delta2
        for x in xs
    ]
    allD3 = [
        x
        for xs in delta3
        for x in xs
    ]
    allD4 = [
        x
        for xs in delta4
        for x in xs
    ]
    allD5 = [
        x
        for xs in delta5
        for x in xs
    ]
    min1 = np.min(allD1); min2 = np.min(allD2); min3 = np.min(allD3); min4 = np.min(allD4); min5 = np.min(allD5)
    max1 = np.max(allD1); max2 = np.max(allD2); max3 = np.max(allD3); max4 = np.max(allD4); max5 = np.max(allD5)

    bins1 = np.logspace(np.log10(min1), np.log10(max1), 80)
    bins2 = np.logspace(np.log10(min2), np.log10(max2), 80)
    bins3 = np.logspace(np.log10(min3), np.log10(max3), 80)
    bins4 = np.logspace(np.log10(min4), np.log10(max4), 80)
    bins5 = np.logspace(np.log10(min5), np.log10(max5), 80)
    # labels = [f"class {i}" for i in range(len(c))]

    """plt.figure(figsize=(10, 6))
    for x in delta1:
        plt.hist(x, bins=bins1, linewidth=1., alpha=0.5, edgecolor="black", color="tab:blue")
    for x in delta2:
        plt.hist(x, bins=bins1, linewidth=1., alpha=0.5, edgecolor="black", color="tab:orange")
    for x in delta3:
        plt.hist(x, bins=bins1, linewidth=1., alpha=0.5, edgecolor="black", color="tab:green")
    for x in delta4:
        plt.hist(x, bins=bins1, linewidth=1., alpha=0.5, edgecolor="black", color="tab:red")
    for x in delta5:
        plt.hist(x, bins=bins1, linewidth=1., alpha=0.5, edgecolor="black", color="tab:purple")
    """
    plt.scatter(range(len(allD1)), allD1, c="tab:blue")
    plt.scatter(range(len(allD2)), allD2, c="tab:orange")
    plt.scatter(range(len(allD3)), allD3, c="tab:green")
    plt.scatter(range(len(allD4)), allD4, c="tab:red")
    plt.xscale("log")
    plt.xlabel(r"$\Delta$", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

#multiPlot()

#singlePlot(10)

def load_dims_for_tau(tau, beta=0.7, F=16, concat=True):
    datas = []
    for ci in range(10):
        pkl = f"/Users/francescoaldoventurelli/qml/projet_one/dictonaries/tau{tau}/class{ci}/probs/fidelity_Dict_class{ci}_beta_{beta}model{F}.pkl"
        with open(pkl, "rb") as f:
            datas.append(pickle.load(f))
    if concat:
        concatenation = np.concatenate([d["dim"] for d in datas], axis=0)
        return concatenation
    else:
        return np.array(datas)


def cumDelta_hist(tau=10, excited_state=1, nbins=50):
    beta = 0.7
    F = 16
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class0/spectrums/fidelity_Dict_class0_beta_{beta}model{F}.pkl",
            "rb") as f0:
        data0 = pickle.load(f0)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class1/spectrums/fidelity_Dict_class1_beta_{beta}model{F}.pkl",
            "rb") as f1:
        data1 = pickle.load(f1)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class2/spectrums/fidelity_Dict_class2_beta_{beta}model{F}.pkl",
            "rb") as f2:
        data2 = pickle.load(f2)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class3/spectrums/fidelity_Dict_class3_beta_{beta}model{F}.pkl",
            "rb") as f3:
        data3 = pickle.load(f3)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class4/spectrums/fidelity_Dict_class4_beta_{beta}model{F}.pkl",
            "rb") as f4:
        data4 = pickle.load(f4)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class5/spectrums/fidelity_Dict_class5_beta_{beta}model{F}.pkl",
            "rb") as f5:
        data5 = pickle.load(f5)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class6/spectrums/fidelity_Dict_class6_beta_{beta}model{F}.pkl",
            "rb") as f6:
        data6 = pickle.load(f6)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class7/spectrums/fidelity_Dict_class7_beta_{beta}model{F}.pkl",
            "rb") as f7:
        data7 = pickle.load(f7)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class8/spectrums/fidelity_Dict_class8_beta_{beta}model{F}.pkl",
            "rb") as f8:
        data8 = pickle.load(f8)
    with open(
            f"/Users/francescoaldoventurelli/qml/projet_one/dictonariesDelta/tau{tau}/class9/spectrums/fidelity_Dict_class9_beta_{beta}model{F}.pkl",
            "rb") as f9:
        data9 = pickle.load(f9)
    data_list = [data0, data1, data2, data3, data4, data5, data6, data7, data8, data9]

    deltas_by_class = []
    dims_by_class = []

    for ci, data in enumerate(data_list):
        spectrums = data["spectrums"]
        dims = np.asarray(data["dim"])

        if len(spectrums) != len(dims):
            raise ValueError(f"Class {ci}: {len(spectrums)} spectrums vs {len(dims)} dims")

        deltas_ci = np.array(
            [float(np.min(E[:, excited_state] - E[:, 0])) for E in spectrums],
            dtype=float
        )

        deltas_by_class.append(deltas_ci)
        dims_by_class.append(dims)

    deltas = np.concatenate(deltas_by_class)
    dims = np.concatenate(dims_by_class)

    #mask = deltas > 0
    #deltas_pos = deltas[mask]
    #dims_pos = dims[mask]

    """if deltas_pos.size == 0:
        raise ValueError("No positive deltas available for log-scale histogram.")"""
    bins = np.logspace(np.log10(deltas.min()), np.log10(deltas.max()), nbins)

    fig, axs = plt.subplots(1, 1, figsize=(5,4))

    unique_dims = np.unique(dims)

    delta = plotDelta(tau, excited_state)
    allDeltas = np.array([x for xs in delta for x in xs], dtype=float)

    #bins = np.logspace(np.log10(dmin), np.log10(dmax), nbins)
    histdim3, _ = np.histogram(deltas[dims == 3], bins=bins)
    histdim5, _ = np.histogram(deltas[dims == 5], bins=bins)
    histdim6, _ = np.histogram(deltas[dims == 6], bins=bins)
    histdim7, _ = np.histogram(deltas[dims == 7], bins=bins)
    histdim8, _ = np.histogram(deltas[dims == 8], bins=bins)

    hist, edges = np.histogram(allDeltas, bins=bins)
    cum = np.cumsum(hist) / hist.sum()
    cum3, cum5, cum6, cum7, cum8 = np.cumsum(histdim3) / histdim3.sum(), np.cumsum(histdim5) / histdim5.sum(), np.cumsum(histdim6) / histdim6.sum(), np.cumsum(histdim7) / histdim7.sum(), np.cumsum(histdim8) / histdim8.sum()

    #plt.bar(edges[1:], cum, color="tab:blue")
    #print("N =", len(allDeltas))
    #print("hist.sum() =", hist.sum())
    #print("cum[-1] =", cum[-1])
    colors = {3: "#233C4B", 5:"#FF7D2D", 6:"#FAC846", 7:"#A0C382", 8:"#5F9B8C"}

    """for d_val in unique_dims:
        axs[1].hist(
            deltas[dims == d_val],
            bins=bins,
            alpha=0.3,
            linewidth=1.5,
            label=fr"$d = 2^{d_val}$",
            color=colors[d_val],
        )"""
    axs.hist(allDeltas, bins=bins, color="#A0C382", alpha=0.75)


    ax2 = axs.twinx()
    #ax3 = axs[1].twinx()
    #axs[0].bar(bins[1:], hist/np.sum(hist), color="skyblue", edgecolor="tab:blue")
    ax2.step(bins[1:], cum, where="pre", color="#D62828", alpha=1, linewidth=1.2, label="Total Cumulative")
    """ax3.step(bins[1:], cum3, where="pre", color="#233C4B", alpha=1, linewidth=1.5, label=r"$d=3$")
    ax3.step(bins[1:], cum5, where="pre", color="#FF7D2D", alpha=1, linewidth=1.5, label=r"$d=5$")
    ax3.step(bins[1:], cum6, where="pre", color="#FAC846", alpha=1, linewidth=1.5, label=r"$d=6$")
    ax3.step(bins[1:], cum7, where="pre", color="#A0C382", alpha=1, linewidth=1.5, label=r"$d=7$")
    ax3.step(bins[1:], cum8, where="pre", color="#5F9B8C", alpha=1, linewidth=1.5, label=r"$d=8$")"""
    #for ax in axs:
    #    ax.tick_params(axis="x", which="both", labelsize=13)
    #    ax.tick_params(axis="y", which="both", labelsize=13)
    axs.tick_params(axis="y", which="both", labelsize=14)
    ax2.tick_params(axis="y", which="both", labelsize=14)
    axs.set_xticks(unique_dims, unique_dims, fontsize=14)
    axs.set_xscale("log")
    #ax3.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_ylabel(r"$\text{Prob}(\Delta_{\min})$", fontsize=16,labelpad=20)
    #axs[1].set_xscale("log")
    axs.set_xlabel(r"$\Delta_{\min}$", fontsize=18)
    #axs[1].set_xlabel(r"$\Delta_{\min}$", fontsize=16)
    axs.set_ylabel("Counts", fontsize=16)

    #axs[1].set_ylabel(r"$\text{Prob}(\Delta_{\min})$", fontsize=16)

    axs.grid(True, which="both", linestyle=":", linewidth=0.5)
    #axs[1].grid(True, which="both", linestyle=":", linewidth=0.5)

    #axs[1].legend(loc="upper left", fontsize=12, frameon=True, edgecolor="k")
    ax2.legend(loc="upper left", fontsize=12, frameon=True, edgecolor="k")

    plt.tight_layout()
    #plt.savefig("/Users/francescoaldoventurelli/qml/projet_one/Deltaditr_Cum.pdf")
    plt.show()



#cumDelta_hist()