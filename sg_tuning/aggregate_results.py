import os
import glob
import json
import statistics


DIR = "./jiant/runs"


def aggregate(task, metric_a, metric_b=None):
    metric_a_list, metric_b_list, durations = [], [], []
    for folder in glob.glob(f"{DIR}/run_{task}_seed*"):
        if not os.path.isfile(f"{folder}/val_metrics.json"):
            continue
        with open(f"{folder}/val_metrics.json") as f:
            metrics = json.load(f)

        metric_a_list.append(metrics[task]["metrics"]["minor"][metric_a])
        if metric_b is not None:
            metric_b_list.append(metrics[task]["metrics"]["minor"][metric_b])

        with open(f"{folder}/time.txt") as f:
            duration = float(f.read().strip())
        durations.append(duration)

    print(task.upper())
    print(f"number of samples: {len(metric_a_list)}")
    if len(metric_a_list) == 0:
        print()
        return

    print(f"{metric_a}: {statistics.mean(metric_a_list)*100:.3f} ± {statistics.stdev(metric_a_list)*100:.3f} %")
    if metric_b is not None:
        print(f"{metric_b}: {statistics.mean(metric_b_list)*100:.3f} ± {statistics.stdev(metric_b_list)*100:.3f} %")
    print(f"time: {statistics.mean(durations):.3f} ± {statistics.stdev(durations):.3f} s")
    print()


aggregate("cb", "avg_f1", "acc")
aggregate("copa", "acc")
aggregate("multirc", "f1", "em")
aggregate("rte", "acc")
aggregate("wic", "acc")
aggregate("wsc", "acc")
aggregate("boolq", "acc")
aggregate("record", "f1", "em")
