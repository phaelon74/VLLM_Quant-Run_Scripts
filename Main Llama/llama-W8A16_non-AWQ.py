import argparse
import yaml
import logging
import sys
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import csv
import re

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 GPTQ quantization on Llama model (no AWQ)."
)
parser.add_argument(
    "model_path",
    type=str,
    help="Path to the source model directory."
)
parser.add_argument(
    "output_path",
    type=str,
    help="Path to the destination directory for saving quantized model."
)
parser.add_argument(
    "recipe_yaml",
    type=str,
    help="Path to the dataset recipe YAML file (contains max_seq_length and dataset config)."
)
parser.add_argument(
    "group_size",
    type=int,
    help="Group size for W8A16 quantization (e.g., 32, 64, 128)."
)

args = parser.parse_args()
model_path = args.model_path
output_path = args.output_path
recipe_yaml_path = args.recipe_yaml
group_size = args.group_size

# =========================
# Setup Logging and Metrics Collection
# =========================
# Create log directory in output path
log_dir = Path(output_path).parent / "quantization_logs"
log_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"quantization_{timestamp}.log"
metrics_csv = log_dir / f"quantization_metrics_{timestamp}.csv"

# Open log file for writing
log_file_handle = open(log_file, 'w', encoding='utf-8')

# Direct stderr wrapper to catch structlog output
class StderrWrapper:
    def __init__(self, log_file_handle, metrics_collector, original_stderr):
        self.log_file_handle = log_file_handle
        self.metrics_collector = metrics_collector
        self.original_stderr = original_stderr
        self.buffer = ""
        
    def write(self, data):
        # Write to original stderr first (so user sees it)
        self.original_stderr.write(data)
        self.original_stderr.flush()
        
        # Handle carriage returns - replace with newline for log file
        log_data = data.replace('\r', '\n')
        
        # Write to log file
        self.log_file_handle.write(log_data)
        self.log_file_handle.flush()
        
        # Buffer and parse complete lines
        self.buffer += log_data
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]  # Keep incomplete line
            for line in lines[:-1]:
                if line.strip() and ('|' in line or 'METRIC' in line or 'Quantizing' in line):
                    self.metrics_collector.parse_log_line(line.strip())
    
    def flush(self):
        self.original_stderr.flush()
        self.log_file_handle.flush()
        
        # Process any remaining buffer
        if self.buffer.strip():
            if '|' in self.buffer or 'METRIC' in self.buffer or 'Quantizing' in self.buffer:
                self.metrics_collector.parse_log_line(self.buffer.strip())
            self.buffer = ""
    
    def fileno(self):
        return self.original_stderr.fileno()
    
    def isatty(self):
        return self.original_stderr.isatty()
    
    def readable(self):
        return False
    
    def writable(self):
        return True
    
    def seekable(self):
        return False

# Metrics collector
class MetricsCollector:
    def __init__(self, csv_path):
        self.metrics = []
        self.csv_path = csv_path
        self.current_module = None  # Track current module being quantized
        self.current_metric = {}  # Accumulate metric data
        
    def parse_log_line(self, line):
        """Parse log line to extract quantization metrics"""
        # Pattern: Quantizing model.layers.X.module_name
        module_match = re.search(r'Quantizing (model\.[\w.]+)', line)
        if module_match:
            # New module started - save previous if complete
            if self.current_module and 'error' in self.current_metric:
                self.metrics.append({
                    'module': self.current_module,
                    'error': self.current_metric.get('error'),
                    'time': self.current_metric.get('time'),
                    'size_mb': self.current_metric.get('size_mb'),
                })
            # Start new module
            self.current_module = module_match.group(1)
            self.current_metric = {}
            return None
        
        # Pattern: METRIC - error X.XX
        error_match = re.search(r'METRIC - error ([\d.]+)', line)
        if error_match:
            self.current_metric['error'] = float(error_match.group(1))
        
        # Pattern: METRIC - time X.XXs
        time_match = re.search(r'METRIC - time ([\d.]+)s', line)
        if time_match:
            self.current_metric['time'] = float(time_match.group(1))
        
        # Pattern: METRIC - Compressed module size: X.XX MB
        size_match = re.search(r'METRIC - Compressed module size: ([\d.]+) MB', line)
        if size_match:
            self.current_metric['size_mb'] = float(size_match.group(1))
            # Size is usually the last metric, save if we have error
            if self.current_module and 'error' in self.current_metric:
                self.metrics.append({
                    'module': self.current_module,
                    'error': self.current_metric.get('error'),
                    'time': self.current_metric.get('time'),
                    'size_mb': self.current_metric.get('size_mb'),
                })
                self.current_metric = {}
        
        return None
    
    def save_csv(self):
        """Save metrics to CSV file"""
        # Save any remaining incomplete metric
        if self.current_module and 'error' in self.current_metric:
            self.metrics.append({
                'module': self.current_module,
                'error': self.current_metric.get('error'),
                'time': self.current_metric.get('time'),
                'size_mb': self.current_metric.get('size_mb'),
            })
        
        if not self.metrics:
            print("WARNING: No metrics collected to save", file=sys.stderr)
            return
            
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['module', 'error', 'time', 'size_mb'])
            writer.writeheader()
            writer.writerows(self.metrics)
        
        print(f"Saved {len(self.metrics)} metrics to {self.csv_path}")
        
        # Print summary
        if self.metrics:
            errors = [m['error'] for m in self.metrics if m['error'] is not None]
            if errors:
                print(f"\n=== Quantization Error Summary ===")
                print(f"Total layers quantized: {len(self.metrics)}")
                print(f"Average error: {sum(errors)/len(errors):.2f}")
                print(f"Min error: {min(errors):.2f}")
                print(f"Max error: {max(errors):.2f}")
                print(f"Layers with error > 10: {sum(1 for e in errors if e > 10)}")
                print(f"Layers with error > 20: {sum(1 for e in errors if e > 20)}")

metrics_collector = MetricsCollector(metrics_csv)

# Wrap stderr to capture all output (including structlog)
original_stderr = sys.stderr
original_stderr_fd = sys.stderr.fileno()

# Create a pipe for FD-level interception
read_fd, write_fd = os.pipe()
saved_stderr_fd = os.dup(original_stderr_fd)
original_stderr_file = os.fdopen(saved_stderr_fd, 'w', encoding='utf-8', buffering=1)

# Store for cleanup
_stderr_cleanup_vars = {
    'write_fd': write_fd,
    'saved_stderr_fd': saved_stderr_fd,
    'original_stderr_fd': original_stderr_fd,
    'original_stderr_file': original_stderr_file
}

# Redirect stderr FD to pipe (catches direct FD writes from structlog)
os.dup2(write_fd, original_stderr_fd)

# Create wrapper for sys.stderr writes
stderr_wrapper = StderrWrapper(log_file_handle, metrics_collector, original_stderr_file)
sys.stderr = stderr_wrapper

# Start thread to read from pipe (catches FD-level writes) and write to both log and stderr
def pipe_reader():
    read_file = os.fdopen(read_fd, 'r', encoding='utf-8', errors='replace', buffering=1)
    while True:
        try:
            line = read_file.readline()
            if not line:
                break
            # Write to original stderr
            original_stderr_file.write(line)
            original_stderr_file.flush()
            # Write to log file (cleaned)
            log_line = line.replace('\r', '\n')
            log_file_handle.write(log_line)
            log_file_handle.flush()
            # Parse for metrics
            if line.strip() and ('|' in line or 'METRIC' in line or 'Quantizing' in line):
                metrics_collector.parse_log_line(line.strip())
        except (OSError, ValueError, BrokenPipeError):
            break
    read_file.close()

pipe_thread = threading.Thread(target=pipe_reader, daemon=True)
pipe_thread.start()

# Custom stdout wrapper to capture stdout output
class TeeOutput:
    def __init__(self, *files):
        self.files = files
        self.original_stdout = sys.stdout
        self.buffer = ""  # Buffer for incomplete lines
        
    def write(self, data):
        # Handle carriage returns (^M) from progress bars
        # Progress bars use \r to overwrite the same line
        # We want to keep the last complete line before \r sequences
        display_data = data
        
        # For log file: replace \r with newline to preserve progress updates
        # But for structured logs (with timestamps), keep them as-is
        if '\r' in data:
            # If this looks like a progress bar (no timestamp), replace \r with \n
            # Otherwise, it's a structured log line, keep it
            if '|' in data and any(x in data for x in ['INFO', 'WARNING', 'ERROR', 'METRIC']):
                # Structured log - keep as-is but replace \r
                log_data = data.replace('\r', '\n')
            else:
                # Progress bar - replace \r with \n
                log_data = data.replace('\r', '\n')
        else:
            log_data = data
        
        # Write to log files
        for f in self.files:
            f.write(log_data)
            f.flush()
        # Write original to stdout (with progress bars)
        self.original_stdout.write(display_data)
        self.original_stdout.flush()
        
        # Buffer and parse complete lines (only parse non-progress-bar lines)
        self.buffer += log_data
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]  # Keep incomplete line in buffer
            for line in lines[:-1]:
                if line.strip() and ('|' in line or 'METRIC' in line or 'Quantizing' in line):
                    metrics_collector.parse_log_line(line)
    
    def flush(self):
        # Process any remaining buffer content
        if self.buffer.strip():
            metrics_collector.parse_log_line(self.buffer)
            self.buffer = ""
        for f in self.files:
            f.flush()
        self.original_stdout.flush()

class TeeError:
    def __init__(self, *files):
        self.files = files
        self.original_stderr = sys.stderr
        self.buffer = ""  # Buffer for incomplete lines
        
    def write(self, data):
        # Handle carriage returns (^M) from progress bars
        # Progress bars use \r to overwrite the same line
        # We want to keep the last complete line before \r sequences
        display_data = data
        
        # For log file: replace \r with newline to preserve progress updates
        # But for structured logs (with timestamps), keep them as-is
        if '\r' in data:
            # If this looks like a progress bar (no timestamp), replace \r with \n
            # Otherwise, it's a structured log line, keep it
            if '|' in data and any(x in data for x in ['INFO', 'WARNING', 'ERROR', 'METRIC']):
                # Structured log - keep as-is but replace \r
                log_data = data.replace('\r', '\n')
            else:
                # Progress bar - replace \r with \n
                log_data = data.replace('\r', '\n')
        else:
            log_data = data
        
        # Write to log files
        for f in self.files:
            f.write(log_data)
            f.flush()
        # Write original to stderr (with progress bars)
        self.original_stderr.write(display_data)
        self.original_stderr.flush()
        
        # Buffer and parse complete lines (only parse non-progress-bar lines)
        self.buffer += log_data
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines[-1]  # Keep incomplete line in buffer
            for line in lines[:-1]:
                if line.strip() and ('|' in line or 'METRIC' in line or 'Quantizing' in line):
                    metrics_collector.parse_log_line(line)
    
    def flush(self):
        # Process any remaining buffer content
        if self.buffer.strip():
            metrics_collector.parse_log_line(self.buffer)
            self.buffer = ""
        for f in self.files:
            f.flush()
        self.original_stderr.flush()

# Custom logging handler to capture llmcompressor structured logs
class StructuredLogHandler(logging.Handler):
    def __init__(self, log_file_handle, metrics_collector):
        super().__init__()
        self.log_file_handle = log_file_handle
        self.metrics_collector = metrics_collector
        
    def emit(self, record):
        # Format the log record - preserve original format from llmcompressor
        msg = record.getMessage()
        
        # If it's a structured log (has timestamp format), preserve it
        if hasattr(record, 'created'):
            # Try to match llmcompressor's format: timestamp | logger | level - message
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            formatted = f"{timestamp}+0000 | {record.name} | {record.levelname} - {msg}"
        else:
            formatted = f"{record.name} | {record.levelname} - {msg}"
        
        # Write to log file
        self.log_file_handle.write(formatted + '\n')
        self.log_file_handle.flush()
        
        # Also write to original stdout so user sees it (but don't double-write through tee)
        if sys.stdout != tee_stdout:
            print(formatted)
        
        # Parse for metrics
        self.metrics_collector.parse_log_line(formatted)

# Store original stdout before replacing
original_stdout = sys.stdout
original_stdout_fd = sys.stdout.fileno()

# Create a pipe for FD-level stdout interception
stdout_read_fd, stdout_write_fd = os.pipe()
saved_stdout_fd = os.dup(original_stdout_fd)
original_stdout_file = os.fdopen(saved_stdout_fd, 'w', encoding='utf-8', buffering=1)

# Redirect stdout FD to pipe (catches direct FD writes from structlog)
os.dup2(stdout_write_fd, original_stdout_fd)

# Store for cleanup
_stdout_cleanup_vars = {
    'write_fd': stdout_write_fd,
    'saved_stdout_fd': saved_stdout_fd,
    'original_stdout_fd': original_stdout_fd,
    'original_stdout_file': original_stdout_file
}

# Start thread to read from stdout pipe
def stdout_pipe_reader():
    read_file = os.fdopen(stdout_read_fd, 'r', encoding='utf-8', errors='replace', buffering=1)
    while True:
        try:
            line = read_file.readline()
            if not line:
                break
            # Write to original stdout
            original_stdout_file.write(line)
            original_stdout_file.flush()
            # Write to log file (cleaned)
            log_line = line.replace('\r', '\n')
            log_file_handle.write(log_line)
            log_file_handle.flush()
            # Parse for metrics
            if line.strip() and ('|' in line or 'METRIC' in line or 'Quantizing' in line):
                metrics_collector.parse_log_line(line.strip())
        except (OSError, ValueError, BrokenPipeError):
            break
    read_file.close()

stdout_pipe_thread = threading.Thread(target=stdout_pipe_reader, daemon=True)
stdout_pipe_thread.start()

# Also create TeeOutput for sys.stdout writes (Python-level)
tee_stdout = TeeOutput(log_file_handle)
# Note: We don't replace sys.stdout since FD-level intercept handles it

# Setup logging to both file and console
# Note: stdout FD is now redirected to pipe, so writes go through our interceptor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(original_stdout_file)  # Use the original stdout (bypasses FD interception)
    ]
)

# Configure llmcompressor logging and add custom handler
llmcompressor_logger = logging.getLogger('llmcompressor')
llmcompressor_logger.setLevel(logging.INFO)
llmcompressor_logger.propagate = False  # Don't propagate to root logger
llmcompressor_logger.addHandler(StructuredLogHandler(log_file_handle, metrics_collector))

compressed_tensors_logger = logging.getLogger('compressed_tensors')
compressed_tensors_logger.setLevel(logging.INFO)
compressed_tensors_logger.propagate = False
compressed_tensors_logger.addHandler(StructuredLogHandler(log_file_handle, metrics_collector))

# Also capture any sub-loggers
for logger_name in ['llmcompressor.modifiers', 'llmcompressor.modifiers.quantization', 
                    'llmcompressor.modifiers.quantization.gptq', 'compress_modules', 'compress']:
    sub_logger = logging.getLogger(logger_name)
    sub_logger.setLevel(logging.INFO)
    sub_logger.propagate = False
    sub_logger.addHandler(StructuredLogHandler(log_file_handle, metrics_collector))

logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("Starting W8A16 GPTQ Quantization")
logger.info("="*80)
logger.info(f"Logging to: {log_file}")
logger.info(f"Metrics will be saved to: {metrics_csv}")

# =========================
# Load Recipe YAML and extract config
# =========================
with open(recipe_yaml_path, 'r') as f:
    recipe_config = yaml.safe_load(f)

# Extract config from calibration_set section
calibration_config = recipe_config.get('calibration_set', {})
MAX_SEQUENCE_LENGTH = calibration_config['max_seq_length']  # Required - fail if missing
SHUFFLE = calibration_config.get('shuffle', True)
SEED = calibration_config.get('seed', 42)
datasets_config = calibration_config.get('datasets', [])

logger.info(f"Loaded recipe from: {recipe_yaml_path}")
logger.info(f"  - max_seq_length: {MAX_SEQUENCE_LENGTH}")
logger.info(f"  - shuffle: {SHUFFLE}")
logger.info(f"  - seed: {SEED}")
logger.info(f"  - group_size: {group_size}")
logger.info(f"  - datasets to load: {len(datasets_config)}")

# =========================
# Model
# =========================
MODEL_ID = model_path

# Load config to check model type
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
logger.info(f"Model config type: {type(config).__name__}")

# Try AutoModelForCausalLM first, fallback to AutoModel for custom models
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
except ValueError as e:
    logger.warning(f"AutoModelForCausalLM failed (likely custom model): {e}")
    logger.info("Attempting AutoModel with trust_remote_code=True...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
    logger.info("Successfully loaded model with AutoModel")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


# =========================
# Dataset Formatters
# =========================
def format_sharegpt(example, columns, tokenizer):
    """Format ShareGPT-style conversations."""
    formatted_messages = []
    
    # Check if first column is system_prompt (for datasets like Gryphe/Sonnet3.5-Charcard-Roleplay)
    if len(columns) >= 2 and 'system' in columns[0].lower():
        system_prompt = example.get(columns[0], '')
        if system_prompt:
            formatted_messages.append({'role': 'system', 'content': str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]
    
    # Get conversation data
    messages = example.get(conv_column, [])
    
    # Handle case where messages is a string (some datasets store JSON strings)
    if isinstance(messages, str):
        try:
            import json
            messages = json.loads(messages)
        except:
            # Not JSON, treat as raw text
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {'text': text}
            return {'text': ''}
    
    # Convert to standard format if needed
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', 'user'))
                content = msg.get('content', msg.get('value', ''))
                # Normalize role names
                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'bot']:
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                if content:  # Only add if there's content
                    formatted_messages.append({'role': role, 'content': str(content)})
            elif isinstance(msg, str):
                # Alternate user/assistant for string lists
                idx = len([m for m in formatted_messages if m['role'] != 'system'])
                role = 'user' if idx % 2 == 0 else 'assistant'
                formatted_messages.append({'role': role, 'content': str(msg)})
    
    if not formatted_messages:
        return {'text': ''}
    
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text}
    except Exception as e:
        # If chat template fails, return empty
        return {'text': ''}


def format_prompt_answer(example, columns, tokenizer):
    """Format prompt/answer pairs (e.g., instruction/response)."""
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    
    prompt = example.get(prompt_col, '')
    answer = example.get(answer_col, '')
    
    messages = [
        {'role': 'user', 'content': str(prompt)},
        {'role': 'assistant', 'content': str(answer)}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    # Try to find messages-like column
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Already in messages format
                    text = tokenizer.apply_chat_template(data, tokenize=False)
                    return {'text': text}
                elif isinstance(data[0], str):
                    # List of strings - alternate user/assistant
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {'text': text}
            elif isinstance(data, str):
                # Single text field
                return {'text': str(data)}
    
    # Fallback: concatenate all columns
    text = ' '.join(str(example.get(col, '')) for col in columns)
    return {'text': text}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = []
    for col in columns:
        if col in example and example[col]:
            texts.append(str(example[col]))
    return {'text': ' '.join(texts)}


FORMATTERS = {
    'sharegpt': format_sharegpt,
    'prompt_answer': format_prompt_answer,
    'chat_completion': format_chat_completion,
    'raw_text': format_raw_text,
}


# =========================
# Load datasets from YAML recipe
# =========================
logger.info("\n=== Loading datasets from recipe ===")
all_datasets = []
total_samples = 0

for ds_config in datasets_config:
    dataset_name = ds_config['dataset']
    split = ds_config.get('split', 'train')
    columns = ds_config.get('columns', [])
    formatter_name = ds_config.get('formatter', 'raw_text')
    num_samples = ds_config.get('num_samples', 10)
    streaming = ds_config.get('streaming', False)
    
    logger.info(f"  Loading: {dataset_name} (split={split}, samples={num_samples}, formatter={formatter_name})")
    
    try:
        # Load dataset
        if streaming:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            # Take samples from streaming dataset
            ds = ds.take(num_samples)
            # Convert to regular dataset
            ds = list(ds)
            from datasets import Dataset
            ds = Dataset.from_list(ds)
        else:
            ds = load_dataset(dataset_name, split=split)
            # Sample from dataset
            n = min(num_samples, len(ds))
            ds = ds.shuffle(seed=SEED).select(range(n))
        
        # Get formatter function
        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        
        # Apply formatter
        ds = ds.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=ds.column_names,
            num_proc=1,  # Use single proc to avoid tokenizer issues
        )
        
        # Filter out empty texts
        ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
        
        all_datasets.append(ds)
        total_samples += len(ds)
        logger.info(f"    -> Loaded {len(ds)} samples")
        
    except Exception as e:
        logger.warning(f"    -> WARNING: Failed to load {dataset_name}: {e}")
        continue

# Concatenate all datasets
if not all_datasets:
    raise ValueError("No datasets were successfully loaded from the recipe!")

ds = concatenate_datasets(all_datasets)
logger.info(f"\n=== Total samples loaded: {total_samples} ===")

# Shuffle combined dataset if requested
if SHUFFLE:
    ds = ds.shuffle(seed=SEED)


# =========================
# Tokenize in batches
# =========================
logger.info("\n=== Tokenizing dataset ===")
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    batched=True,
    remove_columns=ds.column_names,
    num_proc=4,
)

NUM_CALIBRATION_SAMPLES = len(ds)
logger.info(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH})")


# =========================
# Quantization recipe  (W8A16-SYM, non-AWQ GPTQ)
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_quant_args = QuantizationArgs(
    num_bits=8,           # W8 = 8-bit weights
    type="int",
    symmetric=True,       # Symmetric quantization (Marlin-friendly)
    strategy="group",     # Group-wise quantization
    group_size=group_size,  # User-specified group size
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_quant_args,
    input_activations=None,   # A16 = activations untouched (FP16/BF16)
    output_activations=None,
)

recipe = [
    GPTQModifier(
        ignore=["lm_head"],
        config_groups={"group_0": quant_scheme},
        dampening_frac=0.1,  # Standard GPTQ dampening (lower = more aggressive, higher = more stable)
        actorder="static",   # Static ordering - best accuracy with no runtime cost, TP-compatible
        # NOTE: "static" (default) and "group" work with tensor parallelism
        # "dynamic" breaks TP because it requires runtime column reordering via g_idx
        block_size=128,      # Columns processed per pass (default: 128)
    ),
]

# =========================
# Run one-shot compression
# =========================
logger.info("\n=== Running one-shot compression ===")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
)

# =========================
# Quick sanity generation
# =========================
#print("\n\n========== SAMPLE GENERATION ==============")
#dispatch_for_generation(model)
#input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
#output = model.generate(input_ids, max_new_tokens=100)
#print(tokenizer.decode(output[0]))
#print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Cleanup FD interception safely
def safe_cleanup():
    """Safely restore stdout/stderr file descriptors"""
    # Flush all buffers
    try:
        tee_stdout.flush()
    except Exception:
        pass
    try:
        stderr_wrapper.flush()
    except Exception:
        pass
    
    # Close write ends of pipes (causes readers to exit)
    try:
        os.close(_stderr_cleanup_vars['write_fd'])
    except OSError:
        pass
    try:
        os.close(_stdout_cleanup_vars['write_fd'])
    except OSError:
        pass
    
    # Wait for pipe readers to finish
    time.sleep(0.5)
    
    # Restore stderr FD
    try:
        os.dup2(_stderr_cleanup_vars['saved_stderr_fd'], _stderr_cleanup_vars['original_stderr_fd'])
    except OSError:
        pass
    try:
        os.close(_stderr_cleanup_vars['saved_stderr_fd'])
    except OSError:
        pass
    try:
        _stderr_cleanup_vars['original_stderr_file'].close()
    except Exception:
        pass
    
    # Restore stdout FD
    try:
        os.dup2(_stdout_cleanup_vars['saved_stdout_fd'], _stdout_cleanup_vars['original_stdout_fd'])
    except OSError:
        pass
    try:
        os.close(_stdout_cleanup_vars['saved_stdout_fd'])
    except OSError:
        pass
    try:
        _stdout_cleanup_vars['original_stdout_file'].close()
    except Exception:
        pass

safe_cleanup()

# Restore original stdout/stderr objects
sys.stdout = original_stdout
sys.stderr = original_stderr

# Save metrics
metrics_collector.save_csv()

# Close log file
log_file_handle.close()

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
print(f"Log file: {log_file}")
print(f"Metrics CSV: {metrics_csv}")
