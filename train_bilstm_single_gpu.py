"""
Flexible GPU Training Script for BiLSTM Model
Uses accelerate for seamless scaling from 1 GPU to multiple GPUs
Just update num_gpus in config file - no code changes needed!
"""
import yaml
import subprocess
import os
import random
from datetime import datetime

# Load the configuration file
config_file = "configs/bilstm_config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Set environment variables
os.environ["CUDA_HOME"] = config["environment"]["cuda_home"]
os.environ["CUDA_VISIBLE_DEVICES"] = config["environment"]["cuda_visible_devices"]

# Determine number of GPUs from CUDA_VISIBLE_DEVICES
cuda_devices = config["environment"]["cuda_visible_devices"]
num_gpus = len(cuda_devices.split(",")) if cuda_devices else 1

# Set NCCL_DEBUG only for multi-GPU
if num_gpus > 1:
    os.environ["NCCL_DEBUG"] = config["environment"].get("nccl_debug", "INFO")
elif "NCCL_DEBUG" in os.environ:
    del os.environ["NCCL_DEBUG"]

print(f"Using {num_gpus} GPU(s): {cuda_devices}")

# Load training configurations
training_config = config["training"]

# Generate a random port for multi-GPU communication
port = random.randint(25000, 30000)

# Loop through validation folds
for VALIDATION_FOLD in config["validation_folds"]:
    # Generate a random seed for each loop iteration
    seed = random.randint(1, 32767)
    print(f"\n{'='*60}")
    print(f"Starting training for fold {VALIDATION_FOLD}")
    print(f"Seed: {seed}")
    print(f"GPUs: {num_gpus}")
    print(f"{'='*60}\n")

    MODEL_NAME = f"bilstm1-mpware-yuv-fp16-fullfit-seed{seed}-fold{VALIDATION_FOLD}"
    OUTPUT_DIR = f"models/{MODEL_NAME}-maxlen{training_config['max_length']}-lr{training_config['learning_rate']}"
    current_date = datetime.now().strftime("%y%m%d_%H%M")

    # Construct the training command using accelerate (works for 1+ GPUs)
    if num_gpus > 1:
        # Multi-GPU mode
        command = [
            "accelerate",
            "launch",
            "--main_process_port",
            str(port),
            "--multi_gpu",
            "--num_processes",
            str(num_gpus),
            "model.py",
        ]
    else:
        # Single GPU mode (still use accelerate for future scalability)
        command = [
            "accelerate",
            "launch",
            "--num_processes",
            "1",
            "--num_machines",
            "1",
            "--mixed_precision",
            "fp16",
            "model.py",
        ]
    
    # Add common arguments
    command.extend([
        "--output_dir",
        OUTPUT_DIR,
        "--model_path",
        training_config["model_path"],
        "--validation_fold",
        str(VALIDATION_FOLD),
        "--max_length",
        str(training_config["max_length"]),
        "--learning_rate",
        str(training_config["learning_rate"]),
        "--per_device_train_batch_size",
        str(training_config["per_device_train_batch_size"]),
        "--per_device_eval_batch_size",
        str(training_config["per_device_eval_batch_size"]),
        "--num_train_epochs",
        str(training_config["num_train_epochs"]),
        "--save_steps",
        str(training_config["save_steps"]),
        "--o_weight",
        str(training_config["o_weight"]),
        "--seed",
        str(seed),
        "--adv_mode",
        training_config["adv_stop_mode"],
        "--adv_start",
        str(training_config["adv_start"]),
        "--loss",
        training_config["loss"],
        "--smoke_test",
        str(training_config["smoke_test"]),
        "--fullfit",
        str(training_config["fullfit"]),
    ])

    # Execute the command and redirect stdout and stderr
    log_filename = f"logs/bilstm1-fold{VALIDATION_FOLD}-fp16-{current_date}.log"
    print(f"Logging to: {log_filename}\n")
    
    with open(log_filename, "w") as log_file:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1  # Line buffered for real-time output
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()  # Ensure logs are written immediately

    process.stdout.close()
    exit_code = process.wait()
    
    if exit_code == 0:
        print(f"\n✓ Training completed successfully for fold {VALIDATION_FOLD}")
    else:
        print(f"\n✗ Training failed for fold {VALIDATION_FOLD} with exit code {exit_code}")
        print(f"Check log file: {log_filename}")


