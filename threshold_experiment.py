import os
import warnings
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
import FairEM as fem
import workloads as wl
import sys

plt.rcParams["font.family"] = "serif"

warnings.simplefilter(action="ignore", category=FutureWarning)

figure_size = [0, 0]


def save_pandas_csv_if_not_exists(dataframe, outname, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fullname = os.path.join(outdir, outname)
    dataframe.to_csv(fullname, index=False)


def make_acronym(word, delim):
    res = ""
    for spl in word.split(delim):
        res += spl[0].capitalize()
    return res


def run_one_workload(
    model,
    dataset,
    left_sens_attribute,
    right_sens_attribute,
    test_file,
    preds_file,
    single_fairness=True,
    k_combinations=1,
    delimiter=",",
    matching_threshold=0.5,
):
    predictions = pd.read_csv(
        "threshold_experiments/"
        + model.lower()
        + "/"
        + dataset
        + "/"
        + str(matching_threshold)
        + "/"
        + preds_file
        + ".csv"
    ).values.tolist()
    test_file = "data/" + dataset + "/" + test_file

    workload = wl.Workload(
        pd.read_csv(test_file),
        left_sens_attribute,
        right_sens_attribute,
        predictions,
        label_column="label",
        multiple_sens_attr=True,
        delimiter=delimiter,
        single_fairness=single_fairness,
        k_combinations=k_combinations,
    )
    return [workload]


# generates the plots
def create_plots(dataset, experiment):
    if dataset == "iTunes-Amazon":
        models = [
            "DeepMatcher",
            "Ditto",
            "GNEM",
            "HierMatcher",
            "MCAN",
            "svm",
            "rf",
            "nb",
            "lg",
            "ln",
            "dt",
        ]
    else:
        models = [
            "DeepMatcher",
            "Ditto",
            "GNEM",
            "HierMatcher",
            "MCAN",
            "svm",
            "rf",
            "nb",
            "lg",
            "ln",
            "dt",
        ]
    neural = {"DeepMatcher", "Ditto", "GNEM", "HierMatcher", "MCAN"}
    colors = [
        "red",
        "blue",
        "darkorange",
        "green",
        "darkviolet",
        "black",
        "gold",
        "hotpink",
        "gray",
        "lightseagreen",
        "lightgreen",
        "sienna",
        "deepskyblue",
    ]

    dataframes = []
    for model in models:
        df = pd.read_csv(
            "experiments/"
            + dataset
            + "/"
            + str(matching_threshold)
            + "/"
            + model
            + "_results_experiment"
            + experiment
            + ".csv"
        )
        dataframes.append(df)

    sens_attributes = dataframes[0]["sens_attr"].to_list()
    counts = dataframes[0]["counts"].to_list()
    dic = {}
    for val1, val2 in zip(sens_attributes, counts):
        dic.update({val1: val2})
    ##################################################
    # Remove subgroup if the number of occurrences in the training data is less than 10
    idx_to_rm = []
    for idx, val in enumerate(counts):
        if val < 10:
            idx_to_rm.append(idx)

    for idx in sorted(idx_to_rm, reverse=True):
        del sens_attributes[idx]
        del counts[idx]
    ##################################################
    # Remove "other" subgroups from the results
    x = sorted(list(set(sens_attributes)))
    x_rmv = []
    for idx, val in enumerate(x):
        if val.find("other") != -1:
            x_rmv.append(idx)
    for idx in sorted(x_rmv, reverse=True):
        del x[idx]
    ##################################################
    # Remove pairwise cases where left and right subgroups are not identical
    x_rmv = []
    for idx, val in enumerate(x):
        if "|" in val:
            if val.split("|")[0].strip() != val.split("|")[1].strip():
                # x.insert(0, x.pop(idx)) #Uncomment if you need to keep the cases where left and right subgroups are not identical
                # else:
                x_rmv.append(idx)
    for idx in sorted(x_rmv, reverse=True):
        del x[idx]
    ##################################################
    # use these values if you want to show the count of each subgroup in the training data
    dataframes_train = []
    for model in models:
        df_train = pd.read_csv(
            "experiments/"
            + dataset
            + "/"
            + str(matching_threshold)
            + "/"
            + model
            + "_results_experiment"
            + experiment
            + ".csv"
        )
        dataframes_train.append(df_train)

    sens_attributes_train = dataframes_train[0]["sens_attr"].to_list()
    counts_train = dataframes_train[0]["counts"].to_list()
    dic_train = {}
    for val1, val2 in zip(sens_attributes_train, counts_train):
        dic_train.update({val1: val2})
    xx = []
    for val in x:
        if dic_train.get(val) is not None:
            xx.append(
                str(int(dic_train.get(val))) + ":" + val + ":" + str(int(dic.get(val)))
            )
        else:
            xx.append(val + ":" + str(int(dic.get(val))))
    ##################################################
    measures = [
        "accuracy_parity",
        "statistical_parity",
        "true_positive_rate_parity",
        "false_positive_rate_parity",
        "negative_predictive_value_parity",
        "positive_predictive_value_parity",
    ]
    y_axis_ticks = ["AP", "SP", "TPRP/FNRP", "FPRP/TNRP", "NPVP/FORP", "PPVP/FDRP"]

    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ax.set_ylabel("Fairness Measure", fontsize=18)
    ax.set_xlabel("Groups", fontsize=18)
    fig.canvas.draw()
    plt.xlim(0, len(x))
    plt.ylim(0, len(measures))
    xticks = np.arange(0.5, len(x) + 0.5, 1).tolist()
    ax.set_xticks(xticks)
    yticks = np.arange(0.5, len(measures) + 0.5, 1).tolist()
    ax.set_yticks(yticks)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="minor", color="#CCCCCC", linestyle=":")
    ax.set_xticklabels(x, fontsize=10, rotation=0)
    ax.set_yticklabels([a for a in y_axis_ticks], fontsize=12)

    x_offset = [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, 0.4, 0.4, -0.4, -0.4]
    y_offset = [0.0, 0.3, -0.3, 0.0, 0.3, -0.3, 0.0, 0.3, -0.3, 0.4, -0.4, 0.4, -0.4]

    counter_list = []
    for i in range(len(dataframes)):
        df = dataframes[i]
        counter = 0
        for index, row in df.iterrows():
            if row["is_fair"]:
                continue
            if row["sens_attr"] in x:
                x_index = x.index(row["sens_attr"])
                y_index = measures.index(row["measure"])
                counter += 1
                marker = "o" if models[i] in neural else "*"
                plt.plot(
                    x_index + 0.5 + x_offset[i],
                    y_index + 0.5 + y_offset[i],
                    marker=marker,
                    markersize=8,
                    markeredgecolor=colors[i],
                    markerfacecolor=colors[i],
                )
        counter_list.append(counter)

    # customize legend
    handles = []
    for idx in range(len(models)):
        handles.append(
            mpatches.Patch(
                color=colors[idx], label=models[idx] + " " + str(counter_list[idx])
            )
        )

    cross_patch = mlines.Line2D(
        [],
        [],
        color="black",
        marker="*",
        linestyle="None",
        markersize=5,
        label="Non-neural",
    )
    circle_patch = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=5,
        label="Neural",
    )
    handles.extend([cross_patch, circle_patch])
    plt.legend(
        handles=handles,
        fontsize="12",
        bbox_to_anchor=(1, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(
        "experiments/" + dataset + "/" + str(matching_threshold) + "/"
        "/Unfairness_experiment" + experiment + ".png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()

    plt.close()


# Single Fairness
def experiment_one(
    model,
    dataset,
    left_sens_attribute,
    right_sens_attribute,
    threshold,
    matching_threshold,
    single_fairness=True,
    test_file="",
    preds_file="",
):
    print("Computing Single Fairness...")

    workloads = run_one_workload(
        model,
        dataset,
        left_sens_attribute,
        right_sens_attribute,
        test_file,
        preds_file,
        single_fairness=single_fairness,
        matching_threshold=matching_threshold,
    )

    fairEM = fem.FairEM(
        model,
        workloads,
        alpha=0.05,
        full_workload_test=test_file,
        threshold=threshold,
        single_fairness=single_fairness,
    )

    binary_fairness = []
    measures = [
        "accuracy_parity",
        "statistical_parity",
        "true_positive_rate_parity",
        "false_positive_rate_parity",
        "negative_predictive_value_parity",
        "positive_predictive_value_parity",
    ]

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        curr_attr_name = workloads[0].k_combs_to_attr_names[k_comb]
        curr_attr_name = curr_attr_name.replace("Contemporary", "Cont.")
        curr_attr_name = curr_attr_name.replace("Electronic", "Elec.")
        curr_attr_name = curr_attr_name.replace("Left Handed Bat", "Left")
        curr_attr_name = curr_attr_name.replace("Right Handed Bat", "Right")
        curr_attr_name = curr_attr_name.replace("Left-hand bat", "Left")
        curr_attr_name = curr_attr_name.replace("Right-hand bat", "Right")
        if curr_attr_name == "sigmod conference":
            curr_attr_name = curr_attr_name.replace("sigmod conference", "Sigmod")
        if curr_attr_name == "sigmod record":
            curr_attr_name = curr_attr_name.replace("sigmod record", "Sigmod Rec.")
        if curr_attr_name == "vldb":
            curr_attr_name = curr_attr_name.replace("vldb", "VLDB")
        if curr_attr_name == "vldb j.":
            curr_attr_name = curr_attr_name.replace("vldb j.", "VLDB J.")
        if curr_attr_name == "acm trans . database syst .":
            curr_attr_name = curr_attr_name.replace(
                "acm trans . database syst .", "TODS"
            )
        if curr_attr_name == "international conference on management of data":
            curr_attr_name = curr_attr_name.replace(
                "international conference on management of data", "Sigmod"
            )
        if (
            curr_attr_name
            == "the vldb journal -- the international journal on very large data bases"
        ):
            curr_attr_name = curr_attr_name.replace(
                "the vldb journal -- the international journal on very large data bases",
                "VLDB J.",
            )
        if curr_attr_name == "acm sigmod record":
            curr_attr_name = curr_attr_name.replace("acm sigmod record", "Sigmod Rec.")
        if curr_attr_name == "acm transactions on database systems ( tods )":
            curr_attr_name = curr_attr_name.replace(
                "acm transactions on database systems ( tods )", "TODS"
            )
        if curr_attr_name == "very large data bases":
            curr_attr_name = curr_attr_name.replace("very large data bases", "VLDB")

        attribute_names.append(curr_attr_name)

    df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])

    if dataset in ["Compas", "CSRankings"]:
        aggregate = "distribution-division"
    else:
        aggregate = "distribution-subtraction"
    for measure in measures:
        # print("measure", measure)
        temp_df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])
        is_fair, counts = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
        temp_df["measure"] = [measure] * len(is_fair)
        temp_df["sens_attr"] = attribute_names
        temp_df["is_fair"] = is_fair
        temp_df["counts"] = counts
        df = pd.concat([temp_df, df])

    save_pandas_csv_if_not_exists(
        dataframe=df,
        outname=model + "_results_experiment1.csv",
        outdir="experiments/" + dataset + "/" + str(matching_threshold) + "/",
    )


# Pairwise Fairness
def experiment_two(
    model,
    dataset,
    left_sens_attribute,
    right_sens_attribute,
    test_file,
    threshold,
    matching_threshold=0.5,
    single_fairness=False,
    preds_file="",
):
    print("Computing Pairwise Fairness...")
    workloads = run_one_workload(
        model,
        dataset,
        left_sens_attribute,
        right_sens_attribute,
        test_file=test_file,
        single_fairness=single_fairness,
        preds_file=preds_file,
        matching_threshold=matching_threshold,
    )
    fairEM = fem.FairEM(
        model,
        workloads,
        alpha=0.05,
        full_workload_test=test_file,
        threshold=threshold,
        single_fairness=single_fairness,
    )

    binary_fairness = []
    measures = [
        "accuracy_parity",
        "statistical_parity",
        "true_positive_rate_parity",
        "false_positive_rate_parity",
        "negative_predictive_value_parity",
        "positive_predictive_value_parity",
    ]

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        curr_attr_name = workloads[0].k_combs_to_attr_names[k_comb]
        curr_attr_name = curr_attr_name.replace("|", " | ")
        curr_attr_name = curr_attr_name.replace("Contemporary", "Cont.")
        curr_attr_name = curr_attr_name.replace("Electronic", "Elec.")
        curr_attr_name = curr_attr_name.replace("inproceedings", "inproc.")
        curr_attr_name = curr_attr_name.replace("Left Handed Bat", "Left")
        curr_attr_name = curr_attr_name.replace("Left-hand bat", "Left")
        curr_attr_name = curr_attr_name.replace("Right-hand bat", "Right")
        curr_attr_name = curr_attr_name.replace("Right Handed Bat", "Right")
        curr_attr_name = curr_attr_name.replace("ACM TODS", "TODS")
        curr_attr_name = curr_attr_name.replace("SIGMOD Rec.", "SIG Rec.")
        curr_attr_name = curr_attr_name.replace("Hip-Hop/Rap", "HH/Rap")
        curr_attr_name = curr_attr_name.replace("patagonia", "patag.")
        curr_attr_name = curr_attr_name.replace("concave", "conc.")
        curr_attr_name = curr_attr_name.replace("Rap & Hip-Hop", "Rap & HH")
        curr_attr_name = curr_attr_name.replace("African-American", "Afr.-Am.")
        curr_attr_name = curr_attr_name.replace("Caucasian", "Cauc.")
        if curr_attr_name == "sigmod conference":
            curr_attr_name = curr_attr_name.replace("sigmod conference", "Sigmod")
        if curr_attr_name == "sigmod record":
            curr_attr_name = curr_attr_name.replace("sigmod record", "Sigmod Rec.")
        if curr_attr_name == "vldb":
            curr_attr_name = curr_attr_name.replace("vldb", "VLDB")
        if curr_attr_name == "vldb j.":
            curr_attr_name = curr_attr_name.replace("vldb j.", "VLDB J.")
        if curr_attr_name == "acm trans . database syst .":
            curr_attr_name = curr_attr_name.replace(
                "acm trans . database syst .", "TODS"
            )
        if curr_attr_name == "international conference on management of data":
            curr_attr_name = curr_attr_name.replace(
                "international conference on management of data", "Sigmod"
            )
        if (
            curr_attr_name
            == "the vldb journal -- the international journal on very large data bases"
        ):
            curr_attr_name = curr_attr_name.replace(
                "the vldb journal -- the international journal on very large data bases",
                "VLDB J.",
            )
        if curr_attr_name == "acm sigmod record":
            curr_attr_name = curr_attr_name.replace("acm sigmod record", "Sigmod Rec.")
        if curr_attr_name == "acm transactions on database systems ( tods )":
            curr_attr_name = curr_attr_name.replace(
                "acm transactions on database systems ( tods )", "TODS"
            )
        if curr_attr_name == "very large data bases":
            curr_attr_name = curr_attr_name.replace("very large data bases", "VLDB")

        attribute_names.append(curr_attr_name)

        df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])

    if dataset in ["Compas", "CSRankings"]:
        aggregate = "distribution-division"
    else:
        aggregate = "distribution-subtraction"
    for measure in measures:
        temp_df = pd.DataFrame(columns=["measure", "sens_attr", "is_fair"])
        is_fair, counts = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)

        temp_df["measure"] = [measure] * len(is_fair)
        temp_df["sens_attr"] = attribute_names
        temp_df["is_fair"] = is_fair
        temp_df["counts"] = counts

        df = pd.concat([df, temp_df])

    save_pandas_csv_if_not_exists(
        dataframe=df,
        outname=model + "_results_experiment2.csv",
        outdir="experiments/" + dataset + "/" + str(matching_threshold) + "/",
    )


def dataset_experiments(
    dataset, sens_att, test_file, preds_file, threshold, matching_threshold
):  # Threshold specifies the fairness threshold
    if dataset == "iTunes-Amazon":
        models = [
            "DeepMatcher",
            "Ditto",
            "GNEM",
            "HierMatcher",
            "MCAN",
            "svm",
            "rf",
            "nb",
            "lg",
            "ln",
            "dt",
        ]
    else:
        models = [
            "DeepMatcher",
            "Ditto",
            "GNEM",
            "HierMatcher",
            "MCAN",
            "svm",
            "rf",
            "nb",
            "lg",
            "ln",
            "dt",
        ]
    print("===============", dataset, "===============")
    for mod in models:
        print("Model:", mod)
        experiment_one(
            model=mod,
            dataset=dataset,
            left_sens_attribute="left_" + sens_att,
            right_sens_attribute="right_" + sens_att,
            single_fairness=True,
            threshold=threshold,
            matching_threshold=matching_threshold,
            test_file=test_file,
            preds_file=preds_file,
        )
        print("-------------------------------------------------")
    # create_plots(dataset, "1")
    # for mod in models:
    #     print("Model:", mod)
    #     experiment_two(
    #         model=mod,
    #         dataset=dataset,
    #         left_sens_attribute="left_" + sens_att,
    #         right_sens_attribute="right_" + sens_att,
    #         single_fairness=False,
    #         threshold=threshold,
    #         matching_threshold=matching_threshold,
    #         test_file=test_file,
    #         preds_file=preds_file,
    #     )
    #     print("-------------------------------------------------")

    # create_plots(dataset, "2")


dataset = sys.argv[1]
sensitive_attr = sys.argv[2]
threshold = sys.argv[3]
figure_size = [int(sys.argv[4]), int(sys.argv[5])]
matching_threshold = float(sys.argv[6])

dataset_experiments(
    dataset,
    sensitive_attr,
    test_file="test.csv",
    preds_file="preds",
    threshold=float(threshold),
    matching_threshold=matching_threshold,
)
