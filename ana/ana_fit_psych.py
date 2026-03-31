import glob, os
import numpy as np
import pandas as pd
import psignifit as ps 
from psignifit import psigniplot
import matplotlib.pyplot as plt

def fit_psych(csv_file, stem, min_total):
    # Load the file
    df = pd.read_csv(csv_file)
    # Reshape the data for fitting
    df["level"] = df["list_seq_threshold"]
    g = df.groupby("level")["list_seq_hit"].agg(["sum", "count"]).reset_index()
    g = g.rename(columns={"sum": "nCorrect", "count": "nTotal"})
    g = g.sort_values("level")
    g = g[g["nTotal"] > min_total]
    data = g[["level", "nCorrect", "nTotal"]].to_numpy()

    # Fitting parameters for psignifit
    conf = ps.Configuration()
    conf.experiment_type = "nAFC"
    conf.experiment_choices = 2
    conf.sigmoidName = "sigmoid"
    conf.threshPC = 0.707
    conf.stimulus_range = [data[:,0].min(), data[:,0].max()] 

    # Run the fitting
    res = ps.psignifit(data, conf)

    # Save the plot of the result
    psigniplot.plot_psychometric_function(res)
    plt.savefig(path_save_fig + "fit_" + stem + ".png", dpi=200, bbox_inches="tight")
    plt.close()
    # Extract the threshold and confidence interval
    param = res.get_parameter_estimate()
    threshold = param['threshold']
    print('threshold:', threshold)
    confidence_interval = param['width']
    print('width:', confidence_interval)

    return threshold, confidence_interval



if __name__ == '__main__':
    # param
    min_total = 2
    name_user = 's07'
    path_save_fig = "ana/results20260203/" + name_user + '/'

    list_threshold = []
    list_confidence_interval = []
    list_condition = []

    csv_files = sorted(glob.glob(path_save_fig + "ecc*.csv"))
    for csv_file in csv_files:
        stem = os.path.splitext(os.path.basename(csv_file))[0]
        threshold, confidence_interval = fit_psych(csv_file, stem, min_total)
        list_threshold.append(threshold)
        list_confidence_interval.append(confidence_interval)
        list_condition.append(stem)
    # Save the results in a DataFrame
    df_summary = pd.DataFrame({
        "condition": list_condition,
        "threshold": list_threshold,
        "confidence_interval": list_confidence_interval
    })
    df_summary.to_csv(path_save_fig + "summary_thresholds.csv", index=False)
    
    """
    x = np.linspace(data[:,0].min(), data[:,0].max(), 200)
    y = res.proportion_correct(x)  # Fitted curve

    plt.plot(x, y, '-')
    plt.plot(data[:,0], data[:,1] / data[:,2], 'o')
    plt.show()
    """
    print('done')
