#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append("/home/shiraishi.tomohiro/Experiment_for_PP-based_SI")

# print(sys.path)

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

# import mlflow
from tqdm import tqdm

import numpy as np
from sicore import InfiniteLoopError
from source.sfs_norm import SfsSelectiveInferenceNorm


is_tracking = True


class PararellExperiment(metaclass=ABCMeta):
    def __init__(self, num_iter: int, num_results: int, num_worker: int):
        self.num_iter = num_iter
        self.num_results = num_results
        self.num_worker = num_worker

    @abstractmethod
    def iter_experiment(self, args) -> tuple:
        """Run each iteration of the experiment

        Args:
            args (tuple): Tuple of data for each iteration

        Returns:
            tuple: Tuple of results of each iteration
        """
        pass

    def experiment(self, dataset: list) -> list:
        """Execute all iterations of the experiment

        Args:
            dataset (list): List of args for iter_experiment method, the size of which must be equal to num_iter

        Returns:
            list: List of results from iter_experiment method
        """

        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, dataset), total=self.num_iter)
            )
        results = [result for result in results if result is not None]
        return results[: self.num_results]

    @abstractmethod
    def run_experiment(self):
        pass


class SfsSINormExperiment(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        seed: int,
        num_samples: int,
        signal: float,
    ):
        super().__init__(int(num_results * 1.05), num_results, num_worker)
        self.num_results = num_results
        self.seed = seed

        self.num_samples = num_samples
        self.signal = signal

        self.rng = np.random.default_rng(seed=self.seed)

    def iter_experiment(self, args) -> tuple:
        """Run inference on a feature randomly picked from the feature set selected by SFS

        Args:
            args (tuple): Tuple consists of three objects
                          First object is feature matrix
                          Second object is response vector
                          Third object is test index in feature set

        Returns:
            tuple: Tuple consists of each method's tuple of results (oc previous proposed)
                   First object is SelectiveInferenceResult
                   Second object is execution time
        """

        X, y, test_index = args
        si = SfsSelectiveInferenceNorm(X, y, 1, 5, test_index)
        si.construct_eta()

        try:
            res = []

            # over conditioning
            start_time = time.time()
            result = si.inference(over_conditioning=True)
            end_time = time.time()
            oc_results = (result, end_time - start_time)
            res.append(oc_results)

            # exhaustive
            start_time = time.time()
            result = si.inference(exhaustive=True)
            end_time = time.time()
            prev_results = (result, end_time - start_time)
            res.append(prev_results)

            # precision
            for precision in [0.001, 0.01]:
                for search_strategy in ["pi1", "pi2", "pi3"]:
                    start_time = time.time()
                    result = si.inference(
                        termination_criterion="precision",
                        precision=precision,
                        search_strategy=search_strategy,
                    )
                    end_time = time.time()
                    res.append((result, end_time - start_time))

            # decision
            for alpha in [0.05, 0.01]:
                for search_strategy in ["pi1", "pi2", "pi3"]:
                    start_time = time.time()
                    result = si.inference(
                        termination_criterion="decision",
                        significance_level=alpha,
                        search_strategy=search_strategy,
                    )
                    end_time = time.time()
                    res.append((result, end_time - start_time))

            res = tuple(res)

        except InfiniteLoopError:
            return None
        except Exception:
            print("Error")
            return None

        return res

    def run_experiment(self):
        if is_tracking:
            pass
            # mlflow.log_params(self.hyperparameters)

        n = self.num_samples
        p = 10
        k = 5

        X_list = []
        y_list = []
        test_index_list = list(self.rng.choice(k, size=self.num_iter))

        for _ in range(self.num_iter):
            X = self.rng.normal(0, 1, size=(n, p))
            beta = np.zeros(p)
            beta[0 : int(0.5 * p)] = self.signal
            y = X @ beta + self.rng.normal(0, 1, n)
            X_list.append(X)
            y_list.append(y)

        dataset = zip(X_list, y_list, test_index_list)

        results = self.experiment(dataset)
        result_dict = {}

        result_dict["oc_si_results"] = [item[0][0] for item in results]
        result_dict["oc_times"] = [item[0][1] for item in results]
        result_dict["prev_si_results"] = [item[1][0] for item in results]
        result_dict["prev_times"] = [item[1][1] for item in results]

        i = 2
        for precision in [0.001, 0.01]:
            for search_strategy in ["pi1", "pi2", "pi3"]:
                key = f"eps{str(precision)}_{search_strategy}"
                result_dict[key + "_si_results"] = [item[i][0] for item in results]
                result_dict[key + "_times"] = [item[i][1] for item in results]
                i += 1

        for alpha in [0.05, 0.01]:
            for search_strategy in ["pi1", "pi2", "pi3"]:
                key = f"alp{str(alpha)}_{search_strategy}"
                result_dict[key + "_si_results"] = [item[i][0] for item in results]
                result_dict[key + "_times"] = [item[i][1] for item in results]
                i += 1

        self.results = result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=40)
    parser.add_argument("--num_worker", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--signal", type=float, default=0.0)
    args = parser.parse_args()

    print("sfs norm")
    print("num_samples", args.num_samples)
    print("signal", args.signal)

    experiment = SfsSINormExperiment(
        args.num_results,
        args.num_worker,
        args.seed,
        args.num_samples,
        args.signal,
    )

    print("start experiment")
    experiment.run_experiment()
    print("end experiment")

    result_path = "results"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    file_name = f"sfs_norm_seed{args.seed}_n{args.num_samples}_delta{args.signal}.pkl"
    print(file_name)
    print()
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
