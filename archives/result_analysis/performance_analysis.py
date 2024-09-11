import os
import json
import pandas as pd
import matplotlib.pyplot as plt



# 1. 定义存储模型路径的根目录
models_root = "./models"
result_analysis_root = "./result_analysis/results"

# 2. 定义用于存储所有模型性能的DataFrame
performance_data = []


# 3. 从目录中提取各个模型的实验结果
def load_model_performance(model_name, dataset_name, condition):
    result_file = os.path.join(
        result_analysis_root, f"{model_name}_{dataset_name}_{condition}.json"
    )

    # 检查文件是否存在，确保每个实验的结果都被记录
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            performance = json.load(f)
            return performance
    else:
        print(f"Result file not found: {result_file}")
        return None


# 4. 汇总各个模型、数据集、条件下的性能
for model in os.listdir(models_root):
    model_path = os.path.join(models_root, model)
    if os.path.isdir(model_path):
        for dataset in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset)
            if os.path.isdir(dataset_path):
                for condition in os.listdir(dataset_path):
                    condition_path = os.path.join(dataset_path, condition)
                    if os.path.isdir(condition_path):
                        performance = load_model_performance(model, dataset, condition)
                        if performance:
                            print(
                                f"Loaded performance for {model}/{dataset}/{condition}"
                            )
                            performance_data.append(
                                {
                                    "Model": model,
                                    "Dataset": dataset,
                                    "Condition": condition,
                                    "Accuracy": performance["accuracy"],
                                    "Loss": performance["loss"],
                                }
                            )

# 5. 将性能数据转换为 Pandas DataFrame 进行分析
performance_df = pd.DataFrame(performance_data)

# 打印性能数据表格
print(performance_df)


# 6. 可视化分析
def plot_performance_by_model(performance_df):
    plt.figure(figsize=(12, 6))
    for model in performance_df["Model"].unique():
        subset = performance_df[performance_df["Model"] == model]
        plt.plot(subset["Condition"], subset["Accuracy"], label=f"{model}", marker="o")

    plt.title("Model Performance Comparison (Accuracy)")
    plt.xlabel("Condition")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("performance_comparison_by_model.png")
    plt.show()


def plot_performance_by_dataset(performance_df):
    plt.figure(figsize=(12, 6))
    for dataset in performance_df["Dataset"].unique():
        subset = performance_df[performance_df["Dataset"] == dataset]
        plt.plot(
            subset["Condition"], subset["Accuracy"], label=f"{dataset}", marker="o"
        )

    plt.title("Dataset Performance Comparison (Accuracy)")
    plt.xlabel("Condition")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("performance_comparison_by_dataset.png")
    plt.show()


def plot_performance_loss(performance_df):
    plt.figure(figsize=(12, 6))
    for model in performance_df["Model"].unique():
        subset = performance_df[performance_df["Model"] == model]
        plt.plot(subset["Condition"], subset["Loss"], label=f"{model}", marker="o")

    plt.title("Model Loss Comparison")
    plt.xlabel("Condition")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("performance_loss_comparison.png")
    plt.show()


# 按照模型、数据集对比性能
plot_performance_by_model(performance_df)
plot_performance_by_dataset(performance_df)
plot_performance_loss(performance_df)

# 7. 保存结果表格到CSV文件
performance_df.to_csv(
    os.path.join(result_analysis_root, "performance_summary.csv"), index=False
)
