import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import glob

plt.style.use("ggplot")

# /out/ 配下の csv ファイルをすべて読み込み、df としてまとめる
df = pd.concat([pd.read_csv(f) for f in glob.glob("./out/*.csv")])

json = True

df = df[df["is_json_mode"] == json]
df = df[["label", "completion_tokens", "latency"]]

# df = df[df["completion_tokens"] < 2000]

plt.figure(figsize=(16, 12))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

for model in df["label"].unique():
    model_data = df[df["label"] == model]
    colors = {"gpt-3.5-turbo-0125": "lightcoral",
              "gpt-4o-2024-05-13": "lightseagreen"}
    plt.scatter(model_data["completion_tokens"],
                model_data["latency"],
                label=model, c=colors[model], marker="x", s=80)

    # 回帰直線
    X = model_data["completion_tokens"]
    y = model_data["latency"]
    X_const = sm.add_constant(X)  # Adds a constant term to the predictor
    model_reg = sm.OLS(y, X_const).fit()
    intercept, slope = model_reg.params

    X_range = np.linspace(X.min(), X.max(), 100)
    X_range_const = sm.add_constant(X_range)
    y_range = model_reg.predict(X_range_const)
    plt.plot(X_range, y_range, color=colors[model], label=f"{
             model} (y={slope:.2f}x+{intercept:.2f})", alpha=0.5)

plt.xlabel("Tokens", fontsize=16)
plt.ylabel("Latency (ms)", fontsize=16)
plt.title("GPT Models Latency / Tokens", fontsize=20)
plt.legend(fontsize=16)
plt.savefig(f"out/latency_vs_tokens_{"json" if json else "text"}.png", dpi=300)


# TPS plot
plt.figure(figsize=(16, 12))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

df_tps = pd.DataFrame()

for model in df["label"].unique():
    model_data = df[df["label"] == model].copy().reset_index()
    colors = {"gpt-3.5-turbo-0125": "lightcoral",
              "gpt-4o-2024-05-13": "lightseagreen"}

    X = model_data["completion_tokens"]
    y = model_data["latency"]
    X_const = sm.add_constant(X)  # Adds a constant term to the predictor
    model_reg = sm.OLS(y, X_const).fit()
    intercept, slope = model_reg.params
    TTFT = intercept

    model_data["completion_time"] = model_data["latency"] - TTFT
    model_data["TPS"] = model_data["completion_tokens"] / \
        model_data["completion_time"] * 1000

    df_tps[model] = model_data["TPS"]

print(df_tps.describe())

boxplot_style = dict(color="mediumblue", linewidth=2)
df_tps.boxplot(meanline=True, showmeans=True,
               meanprops=dict(color="mediumblue", linewidth=2, linestyle="--"),
               boxprops=boxplot_style,
               whiskerprops=boxplot_style,
               medianprops=boxplot_style,
               capprops=boxplot_style)

plt.xlabel("Model", fontsize=16)
plt.ylabel("TPS [t/s]", fontsize=16)
plt.title("GPT Models TPS", fontsize=20)
plt.savefig(f"out/tps_{"json" if json else "text"}.png", dpi=300)
