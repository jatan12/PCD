# TODO: Assume that one has downloaded google drive folder already
import argparse
import csv
import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.stats

TASK_TO_DOMAIN = {
    "re21": "re",
    "re22": "re",
    "re23": "re",
    "re24": "re",
    "re25": "re",
    "re31": "re",
    "re32": "re",
    "re33": "re",
    "re34": "re",
    "re35": "re",
    "re36": "re",
    "re37": "re",
    "re41": "re",
    "re42": "re",
    "re61": "re",
    "mohopperv2": "morl",
    "moswimmerv2": "morl",
    "mo_hopper_v2": "morl",
    "mo_swimmer_v2": "morl",
    "zdt1": "synthetic",
    "zdt2": "synthetic",
    "zdt3": "synthetic",
    "zdt4": "synthetic",
    "zdt6": "synthetic",
    "omnitest": "synthetic",
    "vlmop1": "synthetic",
    "vlmop2": "synthetic",
    "vlmop3": "synthetic",
    "dtlz1": "synthetic",
    "dtlz7": "synthetic",
    "c10mop1": "monas",
    "c10mop2": "monas",
    "c10mop3": "monas",
    "c10mop4": "monas",
    "c10mop5": "monas",
    "c10mop6": "monas",
    "c10mop7": "monas",
    "c10mop8": "monas",
    "c10mop9": "monas",
    "in1kmop1": "monas",
    "in1kmop2": "monas",
    "in1kmop3": "monas",
    "in1kmop4": "monas",
    "in1kmop5": "monas",
    "in1kmop6": "monas",
    "in1kmop7": "monas",
    "in1kmop8": "monas",
    "in1kmop9": "monas",
    "regex": "scientific",
    "molecule": "scientific",
    "zinc": "scientific",
    "rfp": "scientific",
}

DOMAIN_TO_TASKS = {
    "re": [
        "re21",
        "re22",
        "re23",
        "re24",
        "re25",
        "re31",
        "re32",
        "re33",
        "re34",
        "re35",
        "re36",
        "re37",
        "re41",
        "re42",
        "re61",
    ],
    "morl": ["mo-hopper-v2", "mo-swimmer-v2"],
    "synthetic": [
        "zdt1",
        "zdt2",
        "zdt3",
        "zdt4",
        "zdt6",
        "omnitest",
        "vlmop1",
        "vlmop2",
        "vlmop3",
    ],
    "monas": [
        "c10mop1",
        "c10mop2",
        "c10mop3",
        "c10mop4",
        "c10mop5",
        "c10mop6",
        "c10mop7",
        "c10mop8",
        "c10mop9",
        "in1kmop1",
        "in1kmop2",
        "in1kmop3",
        "in1kmop4",
        "in1kmop5",
        "in1kmop6",
        "in1kmop7",
        "in1kmop8",
        "in1kmop9",
    ],
}

KEY_RENAMES = {
    "hypervolume/D(best)": "hv_d_best",
    "hypervolume/100th": "hv_100th",
    "hypervolume/75th": "hv_75th",
    "hypervolume/50th": "hv_50th",
}

DOMAIN_NAMES = {
    "synthetic": "Synthetic",
    "re": "RE",
    "scientific": "Scientific",
    "morl": "MORL",
    "monas": "MONAS",
}

ALG_NAMES = {
    "paretoflow": "ParetoFlow",
    "d_best": r"$\mathcal{D}$(best)",
    "end2end-vallina": "E2E",
    "end2end-gradnorm": "E2E + GN",
    "end2end-pcgrad": "E2E + PC",
    "multihead-vallina": "MH",
    "multihead-pcgrad": "MH + PC",
    "multihead-gradnorm": "MH + GN",
    "mobo-vallina": "MOBO",
    "multiplemodels-com": "MM + COM",
    "multiplemodels-ict": "MM + ICT",
    "multiplemodels-iom": "MM + IOM",
    "multiplemodels-trimentoring": "MM + TM",
    "multiplemodels-vallina": "MM",
    "pc-diffusion": "PCDiffusion (ours)",
    "pc-diffusion-ref-dir": "PCDiffusion (ref-dir)",
    "pc-diffusion-ref-dir-no-noise": "PCDiffusion (ref-dir, no noise)",
    "pc-diffusion-pruning": "PCDiffusion w Pruning (ours)",
    "pc-diffusion-pruning-ref-dir": "PCDiffusion w Pruning (ref-dir)",
    "pc-diffusion-reweight": "PCDiffusion w reweight (ours)",
    "pc-diffusion-reweight-ref-dir": "PCDiffusion (ours)",
    "pc-diffusion-reweight-ref-dir-high-tau": "PCDiffusion (ours)",
}


TASK_RENAMES = {
    "re21": "RE21",
    "re22": "RE22",
    "re23": "RE23",
    "re24": "RE24",
    "re25": "RE25",
    "re31": "RE31",
    "re32": "RE32",
    "re33": "RE33",
    "re34": "RE34",
    "re35": "RE35",
    "re36": "RE36",
    "re37": "RE37",
    "re41": "RE41",
    "re42": "RE42",
    "re61": "RE61",
    "mohopperv2": "MO-Hopper-v2",
    "moswimmerv2": "MO-Swimmer-v2",
    "mo_hopper_v2": "MO-Hopper-v2",
    "mo_swimmer_v2": "MO-Swimmer-v2",
    "zdt1": "ZDT1",
    "zdt2": "ZDT2",
    "zdt3": "ZDT3",
    "zdt4": "ZDT4",
    "zdt6": "ZDT6",
    "omnitest": "OMNITEST",
    "vlmop1": "VLMOP1",
    "vlmop2": "VLMOP2",
    "vlmop3": "VLMOP3",
    "dtlz1": "DTLZ1",
    "dtlz7": "DTLZ7",
    "c10mop1": "C10/MOP1",
    "c10mop2": "C10/MOP2",
    "c10mop3": "C10/MOP3",
    "c10mop4": "C10/MOP4",
    "c10mop5": "C10/MOP5",
    "c10mop6": "C10/MOP6",
    "c10mop7": "C10/MOP7",
    "c10mop8": "C10/MOP8",
    "c10mop9": "C10/MOP9",
    "in1kmop1": "IN-1K/MOP1",
    "in1kmop2": "IN-1K/MOP2",
    "in1kmop3": "IN-1K/MOP3",
    "in1kmop4": "IN-1K/MOP4",
    "in1kmop5": "IN-1K/MOP5",
    "in1kmop6": "IN-1K/MOP6",
    "in1kmop7": "IN-1K/MOP7",
    "in1kmop8": "IN-1K/MOP8",
    "in1kmop9": "IN-1K/MOP9",
    "regex": "Regex",
    "molecule": "Molecule",
    "zinc": "Zinc",
    "rfp": "RFP",
}


def standardize_task_name(task_name):
    match task_name:
        case "mohopperv2":
            return "mo_hopper_v2"
        case "moswimmerv2":
            return "mo_swimmer_v2"
        case _:
            return task_name


def load_hv_csv(filepath: pathlib.Path):
    with open(filepath, "r") as ifstream:
        reader = csv.DictReader(ifstream)
        rows = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 1

    # Rename the keys
    return {KEY_RENAMES[key]: float(val) for key, val in rows[0].items()}


def variant_to_method(variant):
    match variant:
        case "baseline":
            return "pc-diffusion"
        case "baseline-ref-dir":
            return "pc-diffusion-ref-dir"
        case "baseline-ref-dir-no-noise":
            return "pc-diffusion-ref-dir-no-noise"
        case "pruning-ref-dir":
            return "pc-diffusion-pruning-ref-dir"
        case "reweight-ref-dir":
            return "pc-diffusion-reweight-ref-dir"
        case "reweight-ref-dir-high-tau":
            return "pc-diffusion-reweight-ref-dir-high-tau"
        case "prune" | "pruning":
            return "pc-diffusion-pruning"
        case "reweight":
            return "pc-diffusion-reweight"
        case _:
            assert False, variant


def variant_to_directory(task_dir, variant):
    options = []
    for fp in task_dir.iterdir():
        if fp.name == variant:
            options.append((fp, datetime.datetime(1999, 1, 1)))
        else:
            parts = fp.name.split("_")

            if variant != parts[0]:
                continue
            # check if the path has a date at the end of it
            if len(parts) == 2 and parts[-1].startswith("2025"):
                when = datetime.datetime.strptime(parts[-1], "%Y-%m-%d")
                options.append((fp, when))

    assert len(options) > 0, (
        f"No directories for {variant!r} in {task_dir!s} ({list(task_dir.iterdir())})"
    )

    # Find the newest option
    options = sorted(options, key=lambda x: x[1])
    return options[-1][0].name


def load_moddom_guidance_scales(data_dir: pathlib.Path):
    def load_json(filepath):
        with filepath.open("r") as ifstream:
            payload = json.load(ifstream)
        return payload

    results = []
    for domain_dir in (data_dir / "moddom").iterdir():
        domain = domain_dir.name

        if domain == "scientific":
            variant = "reweight-ref-dir"
        else:
            variant = "guidance-ref-dir"

        print(f" ====== Starting domain {domain} ====== ")
        for task_dir in domain_dir.iterdir():
            task = task_dir.name

            dirname = variant_to_directory(task_dir, variant)

            if not (task_dir / dirname).is_dir():
                print(
                    f"WARNING: Missing guidance-scale ablation for task "
                    f"{task}! Continuing..."
                )
                continue

            print(f"<<< Directory {(task_dir / dirname)!s} >>>")
            # Go through every seed
            for result_dir in (task_dir / dirname).iterdir():
                seed = int(result_dir.name)

                if not (result_dir / "results.json").is_file():
                    print(f"WARNING: {domain}, {task}, {dirname} has no results.json")
                    continue
                task_results = load_json(result_dir / "results.json")
                if domain == "scientific":
                    for scale, hypervolumes in task_results.items():
                        results.append(
                            {
                                "task": task,
                                "domain": domain,
                                "seed": seed,
                                "guidance-scale": float(scale),
                                "d_best": hypervolumes["hv_d_best"],
                                "hv_100th": hypervolumes["hv_100th"],
                                "hv_75th": hypervolumes["hv_75th"],
                                "hv_50th": hypervolumes["hv_50th"],
                            }
                        )
                else:
                    # Otherwise, the results should contain a list of objects,
                    # containing the relevant data
                    for obj in task_results:
                        results.append(
                            {
                                "task": task,
                                "domain": domain,
                                "seed": seed,
                                "guidance-scale": float(obj["guidance_scale"]),
                                "d_best": float(obj["hv_d_best"]),
                                "hv_100th": float(obj["hv_100th"]),
                                "hv_75th": float(obj["hv_75th"]),
                                "hv_50th": float(obj["hv_50th"]),
                            }
                        )

        print(f"=== Domain {domain} done =====")
    return pd.DataFrame(results)


def load_baselines(data_dir: pathlib.Path):
    def _parse_dir_name(dir_name):
        parts = dir_name.split("-")
        model = parts[0].lower()
        train_type = parts[1].lower()
        name = f"{model}-{train_type}"
        task = standardize_task_name(parts[2].lower())
        assert task in TASK_TO_DOMAIN, f"Unknown task {task!r}"
        domain = TASK_TO_DOMAIN[task]
        return {"name": name, "task": task, "domain": domain}

    def _load_results_from_dir(dirpath: pathlib.Path):
        results = []
        for path in dirpath.iterdir():
            task_info = _parse_dir_name(path.name)
            for results_dir_path in path.iterdir():
                parts = results_dir_path.name.split("-")
                assert len(parts) > 2, results_dir_path
                seed = int(parts[2].strip("seed"))

                if not (results_dir_path / "hv_results.csv").is_file():
                    print(f"Skipping {results_dir_path!r}!")
                    continue

                hypervolumes = load_hv_csv(results_dir_path / "hv_results.csv")
                row = {"seed": seed, **task_info, **hypervolumes}
                results.append(row)
            print(f"{task_info['task']=}, {task_info['name']=} done!")
        return results

    print("====== Starting End2end ====")
    end2end_results = _load_results_from_dir(data_dir / "End2End-Results")
    print("====== End2end done ====")

    print("====== Starting MOBO =====")
    mobo_results = _load_results_from_dir(data_dir / "MOBO-Results")

    print("====== MOBO done ======")

    print("====== Starting Multihead =====")
    multihead_results = _load_results_from_dir(data_dir / "MultiHead-Results")
    print("====== Multihead done ======")

    print("====== Starting Multiple models =====")
    multiple_models_results = _load_results_from_dir(
        data_dir / "MultipleModels-Results"
    )
    print("====== Multiple models done ======")
    results = [
        *end2end_results,
        *mobo_results,
        *multihead_results,
        *multiple_models_results,
    ]

    return pd.DataFrame(results)


def load_paretoflow(data_dir: pathlib.Path):
    def _load_hv(dirpath):
        paths = [fp for fp in dirpath.glob("*.json")]
        assert len(paths) == 1, paths
        with open(paths[0], "r") as ifstream:
            payload = json.load(ifstream)
        return {KEY_RENAMES[key]: value for key, value in payload.items()}

    results = []
    for path in (data_dir / "ParetoFlow-Results").iterdir():
        parts = path.name.split("-")
        assert len(parts) == 3, path
        task = parts[1].lower()
        assert task in TASK_TO_DOMAIN, f"Unkown task {task!r}"
        domain = TASK_TO_DOMAIN[task]
        assert "seed" in parts[2], path
        seed = int(parts[2].strip("seed"))
        hypervolumes = _load_hv(path)

        row = {
            "seed": seed,
            "domain": domain,
            "task": task,
            "name": "paretoflow",
            **hypervolumes,
        }
        results.append(row)
    return pd.DataFrame(results)


def load_moddom(data_dir: pathlib.Path, variants: str):
    def variant_to_directory(task_dir, variant):
        options = []
        for fp in task_dir.iterdir():
            if fp.name == variant:
                options.append((fp, datetime.datetime(1999, 1, 1)))
            else:
                parts = fp.name.split("_")
                if variant != parts[0]:
                    continue
                # check if the path has a date at the end of it
                if len(parts) == 2 and parts[-1].startswith("2025"):
                    when = datetime.datetime.strptime(parts[-1], "%Y-%m-%d")
                    options.append((fp, when))

        if len(options) < 1:
            print(
                f"WARNING: No directories for {variant!r} in {task_dir!s} ({list(task_dir.iterdir())})"
            )
            return None

        # Find the newest option
        options = sorted(options, key=lambda x: x[1])
        return options[-1][0].name

    # def variant_to_method(variant):
    #     match variant:
    #         case "baseline":
    #             return "pc-diffusion"
    #         case "baseline-ref-dir":
    #             return "pc-diffusion-ref-dir"
    #         case "baseline-ref-dir-no-noise":
    #             return "pc-diffusion-ref-dir-no-noise"
    #         case "pruning-ref-dir":
    #             return "pc-diffusion-pruning-ref-dir"
    #         case "reweight-ref-dir":
    #             return "pc-diffusion-reweight-ref-dir"
    #         case "prune" | "pruning":
    #             return "pc-diffusion-pruning"
    #         case "reweight":
    #             return "pc-diffusion-reweight"
    #         case _:
    #             assert False, variant
    #
    def load_json(filepath):
        with filepath.open("r") as ifstream:
            payload = json.load(ifstream)
        return payload

    results = []
    for domain_dir in (data_dir / "moddom").iterdir():
        domain = domain_dir.name
        print(f" ====== Starting domain {domain} ====== ")
        for task_dir in domain_dir.iterdir():
            task = task_dir.name

            for variant in variants:
                try:
                    dirname = variant_to_directory(task_dir, variant)
                except Exception:
                    if domain == "scientific":
                        print(f"Could not find {variant} for task={task}. Continuing..")
                        continue

                method = variant_to_method(variant)

                if dirname is None:
                    print(
                        f"WARNING: Missing variant {variant} for task "
                        f"{task}! Continuing..."
                    )
                    continue

                print(f"<<< Variant {variant!r} = {method!r} >>>")
                if not (task_dir / dirname).is_dir():
                    print(
                        f"WARNING: Missing variant {variant} for task "
                        f"{task}! Continuing..."
                    )
                    continue

                print(f"<<< Directory {(task_dir / dirname)!s} >>>")
                # Go through every seed
                for result_dir in (task_dir / dirname).iterdir():
                    seed = int(result_dir.name)

                    if not (result_dir / "results.json").is_file():
                        print(
                            f"WARNING: {domain}, {task}, {dirname} has no results.json"
                        )
                        continue
                    hypervolumes = load_json(result_dir / "results.json")

                    if domain == "scientific":
                        # In sci-design the results contain guidance scales for each of them,
                        # so just pick the one with "2.5"
                        hypervolumes = hypervolumes["2.5"]

                    results.append(
                        {
                            "task": task,
                            "domain": domain,
                            "name": method,
                            "seed": seed,
                            **hypervolumes,
                        }
                    )
        print(f"=== Domain {domain} done =====")
    return pd.DataFrame(results)


def get_per_task_hvs(df, tasks, hv_values="hv_100th"):
    mask = df.task.isin(tasks)
    domain_df = df.loc[mask, :]
    domain_df.rename(columns={"name": "method"}, errors="raise", inplace=True)

    domain_df.loc[:, "task"] = domain_df.task.map(TASK_RENAMES)

    # Add d-best score for each task
    d_best_rows = []
    for (task, seed), task_df in domain_df.groupby(["task", "seed"]):
        domain = task_df.iloc[0]["domain"]
        if task.lower() == "vlmop1":
            d_best = task_df.iloc[0]["hv_d_best"]
        else:
            mask = task_df.method == "paretoflow"
            assert mask.sum() == 1
            d_best = task_df.loc[mask, "hv_d_best"].iloc[0]
        d_best_rows.append(
            {
                "method": "d_best",
                "hv_100th": d_best,
                "hv_75th": d_best,
                "hv_50th": d_best,
                "task": task,
                "seed": seed,
                "hv_d_best": d_best,
                "domain": domain,
            }
        )
    d_best_df = pd.DataFrame(d_best_rows)

    domain_df = pd.concat((domain_df, d_best_df))

    per_task_hv = (
        domain_df.groupby(["task", "method"], as_index=False)[hv_values]
        .agg(["mean", "std"])
        .reset_index(drop=True)
    )

    per_task_hv.loc[:, "value"] = (
        per_task_hv.loc[:, "mean"].apply(lambda x: f"{x:.2f}")
        + r"$\pm$"
        + per_task_hv.loc[:, "std"].apply(lambda x: f"{x:.2f}")
    )

    print(per_task_hv)
    per_task_hv = per_task_hv.pivot(index="method", columns="task", values=["value"])
    per_task_hv = per_task_hv.set_index(per_task_hv.index.map(ALG_NAMES), drop=True)

    print(per_task_hv)
    return per_task_hv


def compute_per_task_hvs(df: pd.DataFrame, output_dir: pathlib.Path, percentile: str):
    # Task groups:
    TASK_GROUPS = {
        "re-1": [
            "re21",
            "re22",
            "re23",
            "re24",
            "re25",
            "re31",
            "re32",
        ],
        "re-2": [
            "re33",
            "re34",
            "re35",
            "re36",
            "re37",
            "re41",
            "re42",
            "re61",
        ],
        "morl": ["mo_hopper_v2", "mo_swimmer_v2"],
        "scientific": ["regex", "molecule", "zinc", "rfp"],
        "synthetic-1": [
            "zdt1",
            "zdt2",
            "zdt3",
            "zdt4",
            "zdt6",
        ],
        "synthetic-2": [
            "dtlz1",
            "dtlz7",
            "omnitest",
            "vlmop1",
            "vlmop2",
            "vlmop3",
        ],
        "monas-1": [
            "c10mop1",
            "c10mop2",
            "c10mop3",
            "c10mop4",
            "c10mop5",
            "c10mop6",
            "c10mop7",
            "c10mop8",
            "c10mop9",
        ],
        "monas-2": [
            "in1kmop1",
            "in1kmop2",
            "in1kmop3",
            "in1kmop4",
            "in1kmop4",
            "in1kmop5",
            "in1kmop6",
            "in1kmop7",
            "in1kmop8",
            "in1kmop9",
        ],
    }

    df.loc[:, "domain"] = df.task.map(TASK_TO_DOMAIN)
    all_domains = list(set(TASK_TO_DOMAIN.values()))
    for key, task_set in TASK_GROUPS.items():
        if key != "scientific":
            continue
        print(f"Staring {key!r}")
        hv_df = get_per_task_hvs(df, task_set, hv_values=percentile)
        s = hv_df.style.apply(highlight_max_mean, props="textbf:--rwrap").apply(
            highlight_second_highest_mean, props="underline:--rwrap;"
        )
        column_format = "l" + "c" * len(hv_df.columns)
        s.to_latex(
            output_dir / f"{key}_{percentile}.tex",
            hrules=True,
            column_format=column_format,
        )
        print(f"{key!r} done!")


def compute_avg_rank_tables(
    df: pd.DataFrame,
    rank_precision: int | None,
    domain_order=["synthetic", "morl", "re", "scientific", "monas"],
):
    # Compute rank dataframe
    dataframes = []
    for key, subdf in df.groupby(by=["task", "domain", "seed"]):
        values = subdf.loc[:, "hv_100th"].tolist()
        algs = subdf.loc[:, "name"].tolist()

        task, domain, seed = key
        if task == "vlmop1":
            d_best = subdf.iloc[0]["hv_d_best"]
        else:
            # Pareto-flow has the correct D-best values
            mask = subdf.name == "paretoflow"
            assert mask.sum() == 1, subdf
            d_best = subdf.loc[mask, "hv_d_best"].iloc[0]
        values.append(d_best)
        algs.append("d_best")

        if rank_precision is None:
            rank_values = np.asarray(values)
        else:
            rank_values = np.asarray(values).round(rank_precision)

        # Negate since the rankdata gives the smallest value the best rank
        ranks = scipy.stats.rankdata(-rank_values, method="dense")
        new_df = pd.DataFrame(
            {
                "rank": ranks,
                "hv_100th": values,
                "name": algs,
                "task": [task] * len(values),
                "domain": [domain] * len(values),
                "seed": [seed] * len(values),
            }
        )

        dataframes.append(new_df)
    rank_df = pd.concat(dataframes)

    # ==== Aggregate rank over tasks in a domain while keeping seed fixed ===
    rank_df.loc[:, "domain"] = rank_df.task.map(TASK_TO_DOMAIN)
    per_domain_rank = rank_df.groupby(["domain", "name", "seed"], as_index=False)[
        "rank"
    ].mean()

    # Aggregate the avg rank over seeds
    per_domain_rank = per_domain_rank.groupby(["domain", "name"], as_index=False)[
        "rank"
    ].agg(["mean", "std"])

    # Create column with standard deviation stuff
    per_domain_rank.loc[:, "value"] = (
        per_domain_rank.loc[:, "mean"].apply(lambda x: f"{x:.2f}")
        + r"$\pm$"
        + per_domain_rank.loc[:, "std"].apply(lambda x: f"{x:.2f}")
    )

    # Pivot the table
    pivot_df = per_domain_rank.drop(columns=["mean", "std"], errors="raise").pivot(
        index="name", columns="domain", values="value"
    )
    pivot_df = pivot_df.set_index(pivot_df.index.map(ALG_NAMES), drop=True)

    # Re order & rename columns
    pivot_df = pivot_df[domain_order]
    pivot_df = pivot_df.rename(columns=DOMAIN_NAMES, errors="raise")

    # Add column with average rank over all tasks
    total_rank = rank_df.groupby(["name", "seed"], as_index=False)["rank"].mean()
    total_rank = total_rank.groupby("name", as_index=False)["rank"].agg(["mean", "std"])
    total_rank.loc[:, "value"] = (
        total_rank.loc[:, "mean"].apply(lambda x: f"{x:.2f}")
        + r"$\pm$"
        + total_rank.loc[:, "std"].apply(lambda x: f"{x:.2f}")
    )
    total_rank.drop(columns=["mean", "std"], errors="raise", inplace=True)
    total_rank.set_index("name", inplace=True)
    total_rank.set_index(total_rank.index.map(ALG_NAMES), inplace=True)
    pivot_df.loc[:, "Avg. rank"] = total_rank.loc[:, "value"]

    return pivot_df


def highlight_second_lowest_mean(s, props):
    indexes = np.arange(s.shape[0])
    mu = []
    for val in s.values:
        if isinstance(val, float):
            mu.append(val)
        else:
            mu.append(float(val.split("$")[0]))
    mu = np.asarray(mu)
    return np.where(indexes == np.argpartition(mu, 2)[1], props, "")


def highlight_second_highest_mean(s, props):
    indexes = np.arange(s.shape[0])
    # mu = np.asarray([float(val.split("$")[0]) for val in s.values])
    mu = []
    for val in s.values:
        if isinstance(val, float):
            mu.append(val)
        else:
            mu.append(float(val.split("$")[0]))
    mu = np.asarray(mu)
    return np.where(indexes == np.argpartition(mu, -2)[-2], props, "")


def highlight_min_mean(s, props):
    indexes = np.arange(s.shape[0])

    mu = []
    for val in s.values:
        if isinstance(val, float):
            mu.append(val)
        else:
            mu.append(float(val.split("$")[0]))
    mu = np.asarray(mu)
    # print(s.values)
    # mu = np.asarray([float(val.split("$")[0]) for val in s.values])
    return np.where(indexes == np.nanargmin(mu), props, "")


def highlight_max_mean(s, props):
    indexes = np.arange(s.shape[0])

    mu = []
    for val in s.values:
        if isinstance(val, float):
            mu.append(val)
        else:
            mu.append(float(val.split("$")[0]))
    mu = np.asarray(mu)
    # mu = np.asarray([float(val.split("$")[0]) for val in s.values])
    return np.where(indexes == np.nanargmax(mu), props, "")


def load_all(input_dir, variants):
    moddom_results = load_moddom(input_dir, variants)
    print(f"Moddom contains {moddom_results.shape[0]} rows!")

    baselines = load_baselines(input_dir)
    print(f"baselines contains {baselines.shape[0]} rows!")

    paretoflow_results = load_paretoflow(input_dir)
    print(f"ParetoFlow contains {paretoflow_results.shape[0]} rows!")
    return pd.concat((baselines, paretoflow_results, moddom_results))


def compute_tables(args):
    match args.action:
        case "create-guidance-df":
            guidance_results = load_moddom_guidance_scales(args.input_dir)
            guidance_results.to_parquet(args.output_path)
        case "create-ablation-df":
            moddom_results = load_moddom(args.input_dir, variants=args.moddom_variant)
            moddom_results.to_parquet(args.output_path)
        case "create-main-df":
            df = load_all(args.input_dir, args.moddom_variant)

        case "create-main-table":
            df = load_all(args.input_dir, args.moddom_variant)
            # Remove tasks where re is zero for baselines
            mask = ~df.task.isin(["re23", "re33", "re35", "re36", "re42"])
            df = df.loc[mask, :]

            rank_table = compute_avg_rank_tables(df, args.rank_precision)
            s = rank_table.style.apply(
                highlight_min_mean, props="textbf:--rwrap"
            ).apply(highlight_second_lowest_mean, props="underline:--rwrap;")
            column_format = "l" + "c" * len(rank_table.columns)
            s.to_latex(args.output_path, hrules=True, column_format=column_format)
        case "create-hv-tables":
            df = load_all(args.input_dir, args.moddom_variant)
            compute_per_task_hvs(df, args.output_path, args.hv_percentile)
        case _:
            assert False, args.action

    # moddom_results = load_moddom(args.input_dir, variants=args.moddom_variant)
    # print(f"Moddom contains {moddom_results.shape[0]} rows!")
    # if args.only_moddom:
    #     df = moddom_results
    # else:
    #     baseline_results = load_baselines(args.input_dir)
    #     print(f"Baseline contain {baseline_results.shape[0]} rows!")
    #
    #     paretoflow_results = load_paretoflow(args.input_dir)
    #     print(f"Pareto-flow contains {paretoflow_results.shape[0]} rows!")
    #
    #     df = pd.concat((baseline_results, paretoflow_results, moddom_results))
    #
    # if args.save_df:
    #     df.to_parquet(args.output_path)
    #     print(f"Stored raw dataframe to {args.output_path!s}")
    #
    # TODO: Make figures for the ablation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        type=str,
        choices=[
            "create-guidance-df",
            "create-main-df",
            "create-ablation-df",
            "create-main-table",
            "create-hv-tables",
        ],
        default="create-main-table",
    )
    parser.add_argument("--input_dir", type=pathlib.Path, required=True)
    parser.add_argument("--rank_precision", type=int, default=None)
    parser.add_argument(
        "--hv_percentile", type=str, choices=["hv_100th", "hv_75th", "hv_50th"]
    )
    parser.add_argument("--output_path", required=True, type=pathlib.Path, default=None)
    # parser.add_argument("--save_df", action="store_true")
    # parser.add_argument("--save_table", action="store_true")
    parser.add_argument("--only_moddom", action="store_true")
    parser.add_argument(
        "--moddom_variant",
        type=str,
        nargs="+",
        default="reweight",
    )
    args = parser.parse_args()

    compute_tables(args)
