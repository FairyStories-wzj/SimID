import re  # 导入正则表达式模块 / Import regex module
import subprocess

# 允许用户输入模式 / Allow user to input mode
mode = input("请输入模式（例如 'easy-filtered' 等）：\nPlease input mode (e.g. 'easy-filtered'): ")

# 配置需要执行的s值 / Configure s values to execute
s_values = [1, 2, 3, 4, 5, 10]

# 输出日志文件 / Output log file
log_file = "predict_results.log"
with open(log_file, 'w') as log:
    log.write("执行记录：\nExecution log:\n")

# 执行预测并获取输出 / Execute prediction and get output
def execute_predict(s_value):
    """
    执行预测命令并返回输出。
    Execute prediction command and return output.
    Args:
        s_value: s参数数值 / value of s parameter
    Returns:
        output: 命令行输出 / command line output
    """
    cmd = f"python predict.py -s={s_value} -pr=10 --mode='{mode}'"
    print(f"正在执行命令：{cmd}\nExecuting command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout
    return output

# 使用正则表达式提取Accuracy和Loss / Extract Accuracy and Loss using regex
def extract_results(output):
    """
    从输出中提取Accuracy和Loss。
    Extract Accuracy and Loss from output.
    Args:
        output: 命令行输出 / command line output
    Returns:
        (accuracy, loss): 提取的准确率和损失 / extracted accuracy and loss
    """
    accuracy = None
    loss = None
    
    # 使用正则表达式提取 Accuracy 和 Loss / Extract using regex
    accuracy_match = re.search(r"Accuracy:\s*([\d.]+)", output)
    loss_match = re.search(r"Predict Loss:\s*([\d.]+)", output)
    
    if accuracy_match:
        accuracy = float(accuracy_match.group(1))  # 提取第一个匹配的浮点值 / Extract first matched float
    
    if loss_match:
        loss = float(loss_match.group(1))  # 提取第一个匹配的浮点值 / Extract first matched float

    return accuracy, loss

# 执行每个s的预测并计算平均值 / Run prediction for each s and calculate average
def run_experiment(s_values):
    """
    对每个s值执行多次预测并记录平均结果。
    Run multiple predictions for each s value and record average results.
    Args:
        s_values: s参数列表 / list of s values
    """
    for s_value in s_values:
        accuracies = []
        losses = []
        print(f"正在执行 s={s_value} 的实验...\nRunning experiment for s={s_value}...")

        # 执行5次预测 / Run prediction 5 times
        for i in range(5):
            output = execute_predict(s_value)
            accuracy, loss = extract_results(output)
            if accuracy is not None and loss is not None:
                accuracies.append(accuracy)
                losses.append(loss)

        # 计算平均值 / Calculate average
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        avg_loss = sum(losses) / len(losses) if losses else None

        # 记录到日志文件 / Record to log file
        with open(log_file, 'a') as log:
            log.write(f"\ns={s_value} 执行5次结果：\nResults for s={s_value} (5 runs):\n")
            log.write(f"Accuracy: {accuracies}, 平均Accuracy: {avg_accuracy}\nAverage Accuracy: {avg_accuracy}\n")
            log.write(f"Predict Loss: {losses}, 平均Loss: {avg_loss}\nAverage Loss: {avg_loss}\n")
            log.write("-" * 50 + "\n")

        print(f"s={s_value} 执行完毕，平均Accuracy: {avg_accuracy}, 平均Loss: {avg_loss}\nDone for s={s_value}, Avg Accuracy: {avg_accuracy}, Avg Loss: {avg_loss}\n")

if __name__ == "__main__":
    run_experiment(s_values)
