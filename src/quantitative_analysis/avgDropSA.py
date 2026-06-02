import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


fs = [6.595467836257309/512, 8.284320605435155/512, 13.01093567251462/512, 57.54196422428621/512]


avgD07 = np.load(os.path.join(os.getcwd(), "avgDrop_beta0.7.npz"), allow_pickle=True)
avgD08 = np.load(os.path.join(os.getcwd(), "avgDrop_beta0.8.npz"), allow_pickle=True)
avgD09 = np.load(os.path.join(os.getcwd(), "avgDrop_beta0.9.npz"), allow_pickle=True)
avgD099 = np.load(os.path.join(os.getcwd(), "avgDrop_beta0.99.npz"), allow_pickle=True)
all_results07 = avgD07["all_results"]
all_results08 = avgD08["all_results"]
all_results09 = avgD09["all_results"]
all_results099 = avgD099["all_results"]


sa_per_class07 = {}; sa_per_class07_std = {}
sa_per_class08 = {}; sa_per_class08_std = {}
sa_per_class09 = {}; sa_per_class09_std = {}
sa_per_class099 = {}; sa_per_class099_std = {}


for entry in all_results07:
    class_name = entry['Class']
    sa_per_class07[class_name] = entry['SA']["mean"]
    sa_per_class07_std[class_name] = entry['SA']["std"]

for entry in all_results08:
    class_name = entry['Class']
    sa_per_class08[class_name] = entry['SA']["mean"]
    sa_per_class08_std[class_name] = entry['SA']["std"]

for entry in all_results09:
    class_name = entry['Class']
    sa_per_class09[class_name] = entry['SA']["mean"]
    sa_per_class09_std[class_name] = entry['SA']["std"]

for entry in all_results099:
    class_name = entry['Class']
    sa_per_class099[class_name] = entry['SA']["mean"]
    sa_per_class099_std[class_name] = entry['SA']["std"]

x = np.arange(0, 10, 1)

avg07 = [sa_per_class07[key] for key in sa_per_class07.keys()]
avg08 = [sa_per_class08[key] for key in sa_per_class08.keys()]
avg09 = [sa_per_class09[key] for key in sa_per_class09.keys()]
avg099 = [sa_per_class099[key] for key in sa_per_class099.keys()]
avg07std = [sa_per_class07_std[key] for key in sa_per_class07_std.keys()]
avg08std = [sa_per_class08_std[key] for key in sa_per_class08_std.keys()]
avg09std = [sa_per_class09_std[key] for key in sa_per_class09_std.keys()]
avg099std = [sa_per_class099_std[key] for key in sa_per_class099_std.keys()]

def expon(x, a, b, c):
    return a*np.exp(-b*x) + c

betaVal = np.arange(4)


def plot1():
    plt.errorbar(x, avg07, avg07std, label=r"$\beta=0.7$", marker="o", markersize=8, linestyle="none",
                 capsize=8, capthick=2, color="#b3cde3", linewidth=2.5)
    plt.errorbar(x, avg08, avg08std, label=r"$\beta=0.8$", marker="s", markersize=8, linestyle="none",
                 capthick=2, capsize=8, color="#8c96c6", linewidth=2.5)
    plt.errorbar(x, avg09, avg09std, label=r"$\beta=0.9$", marker="v", markersize=8, linestyle="none",
                 capthick=2, capsize=8, color="#8856a7", linewidth=2.5)
    plt.legend(fontsize=15, loc="upper right", edgecolor="black",
               frameon=True, fancybox=True)
    plt.xticks(range(10), range(10), fontsize=18)
    plt.xlabel("Class", fontsize=20)
    plt.ylabel("Average Drop %", fontsize=20)
    yavg07 = np.arange(-10, 80, 10)
    plt.hlines(0, -0.2, 9.2, color="red", linestyle="dashed", linewidth=2)
    plt.yticks(yavg07, yavg07, fontsize=20)
    plt.tight_layout()
    #plt.savefig("avgDrop.pdf", bbox_inches="tight", dpi=300)
    plt.show()


avgTot07 = np.asarray(avg07).mean()
avgTot08 = np.asarray(avg08).mean()
avgTot09 = np.asarray(avg09).mean()
avgTot099 = np.asarray(avg099).mean()

avgTot07std = np.asarray(avg07).std()
avgTot08std = np.asarray(avg08).std()
avgTot09std = np.asarray(avg09).std()
avgTot099std = np.asarray(avg099).std()

yval = np.array([avgTot07, avgTot08, avgTot09, avgTot099])

(a,b,c), pcov = curve_fit(expon, betaVal, np.log(yval), p0=[0,0, yval[0]])
xmodel = betaVal       # Note: use numpy!
ymodel = np.exp(expon(xmodel, a, b, c))
b_rounded = np.round(b, 2)
a_rounded = np.round(a, 2)
c_rounded = np.round(c, 2)
print("a", a_rounded, "b", b_rounded, "c", c_rounded)

def plot2():
    fig, axs = plt.subplots(2,1, figsize=(6,8))
    x = np.arange(0, 4, 1)
    axs[0].errorbar(x, [avgTot07, avgTot08, avgTot09, avgTot099], [avgTot07std, avgTot08std, avgTot09std, avgTot099std],
                 color="#8856a7", marker="o", markersize=12, linestyle="none", linewidth=2.5,
                 capsize=8, capthick=2)
    axs[0].set_xlabel(r"$\beta$", fontsize=20)
    axs[1].set_xlabel(r"$\beta$", fontsize=20)
    axs[0].set_ylabel("Average Drop %", fontsize=18)
    axs[1].set_ylabel(r"$\frac{\text{len}(x^*)}{N_f=512}$", fontsize=20)
    axs[0].set_xticks(range(len(x)), [0.7, 0.8, 0.9, 0.99], fontsize=20)
    axs[1].plot(range(len(fs)), fs, color="#b3cde3", linestyle="--", marker="o",
                markersize=12, linewidth=2.5)
    axs[1].set_xticks(range(len(x)), [0.7, 0.8, 0.9, 0.99], fontsize=20)
    axs[0].set_yticks(np.arange(0,31,5), np.arange(0,31,5), fontsize=16)
    axs[1].set_yticks(np.arange(0,1.1,0.25), np.arange(0,1.1,0.25), fontsize=20)
    #axs[1].set_yscale("log")
    #axs[0].plot(xmodel, ymodel, "r--",linewidth=2.8, label=rf"${a_rounded}e^{{{b_rounded:.1f}}}+{c_rounded}$")
    ## same problem, it doesn't round it up to 2 digits
    axs[0].plot(xmodel, ymodel, "r--",linewidth=2.8, label=rf"$2.89e^{{{b_rounded:.2f}}}+{c_rounded}$")
    plt.tight_layout()
    axs[0].grid(axis="both", color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    axs[1].grid(axis="both", color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    axs[0].legend(fontsize=15, loc="upper right", edgecolor="black",
               frameon=True, fancybox=True)

    plt.savefig("average_drop_vs_beta_and_solution_length_vs_beta.pdf", bbox_inches="tight",
                dpi=300)
    plt.show()


plot2()
