import pickle

import numpy as np
from scipy.stats import norm
from sicore import type1_error_rate

import matplotlib.pyplot as plt

import os

plt.rcParams.update({"font.size": 16})

# Unique
if not os.path.exists(path := "images/sfs_norm_images"):
    os.mkdir(path)


# Unique
def file_name(seed, num_samples, signal):
    return f"sfs_norm_seed{seed}_n{num_samples}_delta{signal}.pkl"


def open_file(seed=0, num_samples=200, signal=0.0):
    with open("results/" + file_name(seed, num_samples, signal), "rb") as f:
        results_dict = pickle.load(f)
    return results_dict


def get_key(method="oc", eps=None, alp=None, strategy="pi3"):
    if method == "oc":
        key = "oc"
    elif method == "prev":
        key = "prev"
    else:
        if eps is not None:
            key = f"eps{str(eps)}_{strategy}"
        elif alp is not None:
            key = f"alp{str(alp)}_{strategy}"
    return key


def get_results(results_dict, method="oc", eps=None, alp=None, strategy="pi3"):
    key = get_key(method, eps, alp, strategy)
    return results_dict[key + "_si_results"]


def get_times(results_dict, method="oc", eps=None, alp=None, strategy="pi3"):
    key = get_key(method, eps, alp, strategy)
    return np.mean(results_dict[key + "_times"])


def get_prs(results_dict, method="oc", eps=None, alp=None, strategy="pi3", base=0.05):
    results = get_results(results_dict, method, eps, alp, strategy)
    p_list = np.array([result.p_value for result in results])
    if eps is not None:
        p_list = np.array([result.sup_p for result in results])
    return np.mean(p_list <= base)


def get_counts(results_dict, method="oc", eps=None, alp=None, strategy="pi3"):
    results = get_results(results_dict, method, eps, alp, strategy)
    counts = [result.search_count for result in results]
    return np.mean(counts)


def get_count_list(results_dict, method="oc", eps=None, alp=None, strategy="pi3"):
    results = get_results(results_dict, method, eps, alp, strategy)
    counts = [result.search_count for result in results]
    return counts


def plot_fpr(alp=0.05, ymax=0.4):
    samples_list = [100, 200, 300, 400]

    oc_fpr_list = []
    naive_fpr_list = []
    prev_fpr_list = []
    prop_eps_high_fpr_list = []
    prop_eps_low_fpr_list = []
    prop_alp_fpr_list = []

    for samples in samples_list:
        results_dict = open_file(num_samples=samples)

        oc_fpr_list.append(get_prs(results_dict, method="oc", base=alp))

        oc_results = get_results(results_dict, method="oc")
        naive_fpr_list.append(
            type1_error_rate(
                [2 * norm.cdf(-np.abs(result.stat)) for result in oc_results], alp
            )
        )

        prev_fpr_list.append(get_prs(results_dict, method="prev", base=alp))

        prop_eps_high_fpr_list.append(
            get_prs(results_dict, method="prop", eps=0.001, base=alp)
        )
        prop_eps_low_fpr_list.append(
            get_prs(results_dict, method="prop", eps=0.01, base=alp)
        )
        prop_alp_fpr_list.append(
            get_prs(results_dict, method="prop", alp=alp, base=alp)
        )

    _, ax = plt.subplots()

    # ax.set_title(f"Type I Error Rate (significance level is {alp})")
    ax.plot(samples_list, prev_fpr_list, label="exhaustive", marker="x")
    ax.plot(
        samples_list,
        prop_eps_high_fpr_list,
        label="proposed(precision 0.1)",
        marker="x",
    )
    ax.plot(
        samples_list, prop_eps_low_fpr_list, label="proposed(precision 1.0)", marker="x"
    )
    ax.plot(
        samples_list, prop_alp_fpr_list, label=f"proposed(decision {alp})", marker="x"
    )

    ax.plot(samples_list, oc_fpr_list, label="OC", marker="x")
    ax.plot(samples_list, naive_fpr_list, label="naive", marker="x")

    ax.plot(
        samples_list,
        alp * np.ones_like(samples_list),
        color="red",
        linestyle="--",
        lw=0.8,
    )
    ax.legend()
    ax.set_ylim([0, ymax])
    ax.set_xticks(samples_list)

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Type I Error rate")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/fpr_alpha{alp}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_tpr(alp=0.05):
    signal_list = [0.1, 0.2, 0.3, 0.4]

    oc_tpr_list = []
    prev_tpr_list = []
    prop_eps_high_tpr_list = []
    prop_eps_low_tpr_list = []
    prop_alp_tpr_list = []

    for signal in signal_list:
        results_dict = open_file(signal=signal)

        oc_tpr_list.append(get_prs(results_dict, method="oc", base=alp))
        prev_tpr_list.append(get_prs(results_dict, method="prev", base=alp))

        prop_eps_high_tpr_list.append(
            get_prs(results_dict, method="prop", eps=0.001, base=alp)
        )
        prop_eps_low_tpr_list.append(
            get_prs(results_dict, method="prop", eps=0.01, base=alp)
        )
        prop_alp_tpr_list.append(
            get_prs(results_dict, method="prop", alp=alp, base=alp)
        )

    _, ax = plt.subplots()

    # ax.set_title(f"Power (significance level is {alp})")
    ax.plot(signal_list, prev_tpr_list, label="exhaustive", marker="x")
    ax.plot(
        signal_list, prop_eps_high_tpr_list, label="proposed(precision 0.1)", marker="x"
    )
    ax.plot(
        signal_list, prop_eps_low_tpr_list, label="proposed(precision 1.0)", marker="x"
    )
    ax.plot(
        signal_list, prop_alp_tpr_list, label=f"proposed(decision {alp})", marker="x"
    )

    ax.plot(signal_list, oc_tpr_list, label="OC", marker="x")

    ax.legend()
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(signal_list)

    ax.set_xlabel("signal")
    ax.set_ylabel("Power")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/tpr_alpha{alp}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_time_null():
    samples_list = [100, 200, 300, 400]

    prev_time_list = []
    prop_alp_high_time_list = []
    prop_alp_low_time_list = []
    prop_eps_high_time_list = []
    prop_eps_low_time_list = []

    for samples in samples_list:
        results_dict = open_file(num_samples=samples)
        prev_time_list.append(np.mean(get_times(results_dict, method="prev")))

        strategy = "pi3"

        prop_eps_high_time_list.append(
            get_times(results_dict, method="prop", eps=0.001, strategy=strategy)
        )
        prop_eps_low_time_list.append(
            get_times(results_dict, method="prop", eps=0.01, strategy=strategy)
        )

        prop_alp_high_time_list.append(
            get_times(results_dict, method="prop", alp=0.05, strategy=strategy)
        )
        prop_alp_low_time_list.append(
            get_times(results_dict, method="prop", alp=0.01, strategy=strategy)
        )

    _, ax = plt.subplots()

    # ax.set_title("Execution Time (under null)")
    ax.plot(samples_list, prev_time_list, label="exhaustive", marker="x")

    ax.plot(
        samples_list,
        prop_eps_high_time_list,
        label="proposed(precision 0.1)",
        marker="x",
    )
    ax.plot(
        samples_list,
        prop_eps_low_time_list,
        label="proposed(precision 1.0)",
        marker="x",
    )
    ax.plot(
        samples_list,
        prop_alp_high_time_list,
        label="proposed(decision 0.05)",
        marker="x",
    )
    ax.plot(
        samples_list,
        prop_alp_low_time_list,
        label="proposed(decision 0.01)",
        marker="x",
    )

    ax.legend()
    ax.set_xticks(samples_list)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("time")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/time_null.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_time_alternative():
    signal_list = [0.1, 0.2, 0.3, 0.4]

    prev_time_list = []
    prop_alp_high_time_list = []
    prop_alp_low_time_list = []
    prop_eps_high_time_list = []
    prop_eps_low_time_list = []

    for signal in signal_list:
        results_dict = open_file(signal=signal)
        prev_time_list.append(np.mean(get_times(results_dict, method="prev")))

        prop_eps_high_time_list.append(
            get_times(results_dict, method="prop", eps=0.001)
        )
        prop_eps_low_time_list.append(get_times(results_dict, method="prop", eps=0.01))

        prop_alp_high_time_list.append(get_times(results_dict, method="prop", alp=0.05))
        prop_alp_low_time_list.append(get_times(results_dict, method="prop", alp=0.01))

    _, ax = plt.subplots()

    # ax.set_title("Execution Time (under alternative)")
    ax.plot(signal_list, prev_time_list, label="exhaustive", marker="x")

    ax.plot(
        signal_list,
        prop_eps_high_time_list,
        label="proposed(precision 0.1)",
        marker="x",
    )
    ax.plot(
        signal_list,
        prop_eps_low_time_list,
        label="proposed(precision 1.0)",
        marker="x",
    )
    ax.plot(
        signal_list,
        prop_alp_high_time_list,
        label="proposed(decision 0.05)",
        marker="x",
    )
    ax.plot(
        signal_list,
        prop_alp_low_time_list,
        label="proposed(decision 0.01)",
        marker="x",
    )

    ax.legend()
    ax.set_xticks(signal_list)
    ax.set_xlabel("signal")
    ax.set_ylabel("time")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/time_alternative.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_count_under_null_compare_criteria():
    samples_list = [100, 200, 300, 400]

    prev_count_list = []
    prop_alp_high_count_list = []
    prop_alp_low_count_list = []
    prop_eps_high_count_list = []
    prop_eps_low_count_list = []

    for samples in samples_list:
        results_dict = open_file(num_samples=samples)
        prev_count_list.append(np.mean(get_counts(results_dict, method="prev")))

        prop_eps_high_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001)
        )
        prop_eps_low_count_list.append(
            get_counts(results_dict, method="prop", eps=0.01)
        )

        prop_alp_high_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05)
        )
        prop_alp_low_count_list.append(
            get_counts(results_dict, method="prop", alp=0.01)
        )

    _, ax = plt.subplots()

    # ax.set_title("Search Count (compare criteria under null)")
    ax.plot(samples_list, prev_count_list, label="exhaustive", marker="x")

    ax.plot(
        samples_list,
        prop_eps_high_count_list,
        label="proposed(precision 0.1)",
        marker="x",
    )
    ax.plot(
        samples_list,
        prop_eps_low_count_list,
        label="proposed(precision 1.0)",
        marker="x",
    )

    ax.plot(
        samples_list,
        prop_alp_high_count_list,
        label="proposed(decision 0.05)",
        marker="x",
    )
    ax.plot(
        samples_list,
        prop_alp_low_count_list,
        label="proposed(decision 0.01)",
        marker="x",
    )

    ax.legend()
    ax.set_xticks(samples_list)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("count")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/count_null_criteria.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_count_under_null_compare_strategy():
    samples_list = [100, 200, 300, 400]

    prev_count_list = []
    prop_eps_pi2_count_list = []
    prop_eps_pi3_count_list = []
    prop_eps_pi1_count_list = []
    prop_alp_pi2_count_list = []
    prop_alp_pi3_count_list = []
    prop_alp_pi1_count_list = []

    for samples in samples_list:
        results_dict = open_file(num_samples=samples)
        prev_count_list.append(np.mean(get_counts(results_dict, method="prev")))

        prop_eps_pi3_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001, strategy="pi3")
        )
        prop_eps_pi2_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001, strategy="pi2")
        )
        prop_eps_pi1_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001, strategy="pi1")
        )

        prop_alp_pi3_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05, strategy="pi3")
        )
        prop_alp_pi2_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05, strategy="pi2")
        )
        prop_alp_pi1_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05, strategy="pi1")
        )

    _, ax = plt.subplots()

    # ax.set_title("Search Count (compare strategy under null)")
    ax.plot(samples_list, prev_count_list, label="exhaustive", marker="x")

    ax.plot(
        samples_list,
        prop_eps_pi1_count_list,
        label=r"proposed(precision 0.1) with $\pi_1$",
        marker=">",
    )
    ax.plot(
        samples_list,
        prop_eps_pi2_count_list,
        label=r"proposed(precision 0.1) with $\pi_2$",
        marker="^",
    )
    ax.plot(
        samples_list,
        prop_eps_pi3_count_list,
        label=r"proposed(precision 0.1) with $\pi_3$",
        marker="<",
    )

    ax.plot(
        samples_list,
        prop_alp_pi1_count_list,
        label=r"proposed(decision 0.05) with $\pi_1$",
        marker=">",
    )
    ax.plot(
        samples_list,
        prop_alp_pi2_count_list,
        label=r"proposed(decision 0.05) with $\pi_2$",
        marker="^",
    )
    ax.plot(
        samples_list,
        prop_alp_pi3_count_list,
        label=r"proposed(decision 0.05) with $\pi_3$",
        marker="<",
    )

    ax.legend()
    ax.set_xticks(samples_list)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("count")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/count_null_strategy.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_count_under_alternative_compare_criteria():
    signal_list = [0.1, 0.2, 0.3, 0.4]

    prev_count_list = []
    prop_alp_high_count_list = []
    prop_alp_low_count_list = []
    prop_eps_high_count_list = []
    prop_eps_low_count_list = []

    for signal in signal_list:
        results_dict = open_file(signal=signal)
        prev_count_list.append(np.mean(get_counts(results_dict, method="prev")))

        prop_eps_high_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001)
        )
        prop_eps_low_count_list.append(
            get_counts(results_dict, method="prop", eps=0.01)
        )

        prop_alp_high_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05)
        )
        prop_alp_low_count_list.append(
            get_counts(results_dict, method="prop", alp=0.01)
        )

    _, ax = plt.subplots()

    # ax.set_title("Search Count (compare criteria under alternative)")
    ax.plot(signal_list, prev_count_list, label="exhaustive", marker="x")

    ax.plot(
        signal_list,
        prop_eps_high_count_list,
        label="proposed(precision 0.1)",
        marker="x",
    )
    ax.plot(
        signal_list,
        prop_eps_low_count_list,
        label="proposed(precision 1.0)",
        marker="x",
    )

    ax.plot(
        signal_list,
        prop_alp_high_count_list,
        label="proposed(decision 0.05)",
        marker="x",
    )
    ax.plot(
        signal_list,
        prop_alp_low_count_list,
        label="proposed(decision 0.01)",
        marker="x",
    )

    ax.legend()
    ax.set_xticks(signal_list)
    ax.set_xlabel("signal")
    ax.set_ylabel("count")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/count_alternative_criteria.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_count_under_alternative_compare_strategy():
    signal_list = [0.1, 0.2, 0.3, 0.4]

    prev_count_list = []
    prop_eps_pi2_count_list = []
    prop_eps_pi3_count_list = []
    prop_eps_pi1_count_list = []
    prop_alp_pi2_count_list = []
    prop_alp_pi3_count_list = []
    prop_alp_pi1_count_list = []

    for signal in signal_list:
        results_dict = open_file(signal=signal)
        prev_count_list.append(np.mean(get_counts(results_dict, method="prev")))

        prop_eps_pi3_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001, strategy="pi3")
        )
        prop_eps_pi2_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001, strategy="pi2")
        )
        prop_eps_pi1_count_list.append(
            get_counts(results_dict, method="prop", eps=0.001, strategy="pi1")
        )

        prop_alp_pi3_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05, strategy="pi3")
        )
        prop_alp_pi2_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05, strategy="pi2")
        )
        prop_alp_pi1_count_list.append(
            get_counts(results_dict, method="prop", alp=0.05, strategy="pi1")
        )

    _, ax = plt.subplots()

    # ax.set_title("Search Count (compare strategy under alternative)")
    ax.plot(signal_list, prev_count_list, label="exhaustive", marker="x")

    ax.plot(
        signal_list,
        prop_eps_pi1_count_list,
        label=r"proposed(precision 0.1) with $\pi_1$",
        marker=">",
    )
    ax.plot(
        signal_list,
        prop_eps_pi2_count_list,
        label=r"proposed(precision 0.1) with $\pi_2$",
        marker="^",
    )
    ax.plot(
        signal_list,
        prop_eps_pi3_count_list,
        label=r"proposed(precision 0.1) with $\pi_3$",
        marker="<",
    )

    ax.plot(
        signal_list,
        prop_alp_pi1_count_list,
        label=r"proposed(decision 0.05) with $\pi_1$",
        marker=">",
    )
    ax.plot(
        signal_list,
        prop_alp_pi2_count_list,
        label=r"proposed(decision 0.05) with $\pi_2$",
        marker="^",
    )
    ax.plot(
        signal_list,
        prop_alp_pi3_count_list,
        label=r"proposed(decision 0.05) with $\pi_3$",
        marker="<",
    )

    ax.legend()
    ax.set_xticks(signal_list)
    ax.set_xlabel("signal")
    ax.set_ylabel("count")

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/count_alternative_strategy.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_count_box(num_samples=200, signal=0.0):
    results_dict = open_file(num_samples=num_samples, signal=signal)
    counts_list = []

    counts_list.append(get_count_list(results_dict, method="prev"))

    counts_list.append(
        get_count_list(results_dict, method="prop", eps=0.001, strategy="pi1")
    )
    counts_list.append(
        get_count_list(results_dict, method="prop", eps=0.001, strategy="pi2")
    )
    counts_list.append(
        get_count_list(results_dict, method="prop", eps=0.001, strategy="pi3")
    )

    counts_list.append(
        get_count_list(results_dict, method="prop", alp=0.05, strategy="pi1")
    )
    counts_list.append(
        get_count_list(results_dict, method="prop", alp=0.05, strategy="pi2")
    )
    counts_list.append(
        get_count_list(results_dict, method="prop", alp=0.05, strategy="pi3")
    )

    _, ax = plt.subplots()

    # ax.set_title(f"Search Count (sample size {num_samples} and signal {signal})")
    step = 1.1
    ax.boxplot(counts_list, showmeans=True, positions=np.arange(1, 1 + step * 7, step))

    xticklabels = [
        "parametric",
        "precision\npi1",
        "precision\npi2",
        "precision\npi3",
        "decision\npi1",
        "decision\npi2",
        "decision\npi3",
    ]
    ax.set_xticklabels(xticklabels)

    # Unique
    plt.savefig(
        f"images/sfs_norm_images/box_count_samples{num_samples}_signal{signal}.pdf",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


if __name__ == "__main__":
    plot_fpr(0.05, 0.4)
    plot_fpr(0.01, 0.08)
    plot_tpr(0.05)
    plot_tpr(0.01)

    plot_time_null()
    plot_time_alternative()

    plot_count_under_null_compare_criteria()
    plot_count_under_null_compare_strategy()

    plot_count_under_alternative_compare_criteria()
    plot_count_under_alternative_compare_strategy()

    for n in [100, 200, 300, 400]:
        continue
        plot_count_box(num_samples=n)
    for signal in [0.1, 0.2, 0.3, 0.4]:
        continue
        plot_count_box(signal=signal)
