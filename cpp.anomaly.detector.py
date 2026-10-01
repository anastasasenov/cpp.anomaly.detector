#!/usr/bin/env python3

# C/C++ Anomaly Detection Pipeline ( P1, P2, P3, P4 )

import os
import re
import math
import random
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
import clang.cindex
import warnings

# warnings
warnings.filterwarnings("ignore", message="mtime may not be reliable on this filesystem")

# globals
S_ITEM = "snippet"
F_THRESHOLD = "threshold.txt"

# P1: Data/AST Parser 
class CParser:

    def __init__(self, std_flag: str):
        if clang is None:
            raise ImportError("Required clang package!")
        self.std_flag = std_flag
        try:
            self.index = clang.cindex.Index.create()
        except Exception as e:
            logging.warning(f"Failed initialize Clang index: {e}.")
            self.index = None

    def parse_file(
        self,
        filepath: str) -> List[Dict[str, Any]]:
        if not self.index:
            logging.error("Clang index initialization failed.")
            return []
        
        extracted_elements = []
        try:
            args = [f'-std={self.std_flag}', '-w']
            opts = clang.cindex.TranslationUnit.PARSE_INCOMPLETE
            translation_unit = self.index.parse(filepath, args=args, options=opts)
        except Exception as e:
            # continue on the next file
            logging.warning(f"Failed to parse {filepath}: {e}")
            return []

        def _traverse(node: clang.cindex.Cursor, current_file: str):
            # Only process nodes from the target file (exclude system headers)
            if node.location.file and os.path.abspath(node.location.file.name) == os.path.abspath(current_file):
                kind = node.kind
                # Target constructs: classes, functions, methods, structs, templates, enums
                target_kinds = {
                    clang.cindex.CursorKind.CLASS_DECL,
                    clang.cindex.CursorKind.STRUCT_DECL,
                    clang.cindex.CursorKind.FUNCTION_DECL,
                    clang.cindex.CursorKind.CXX_METHOD,
                    clang.cindex.CursorKind.ENUM_DECL,
                    clang.cindex.CursorKind.CLASS_TEMPLATE,
                    clang.cindex.CursorKind.FUNCTION_TEMPLATE
                }
                
                if kind in target_kinds and node.spelling:
                    snippet = self._m_get_node_source(node, current_file)
                    # ASCII sanitization & reporting non-ASCII names
                    name = node.spelling
                    if not all(ord(char) < 128 for char in name):
                        logging.warning(f"Non-ASCII in: '{name}' at {current_file}:{node.location.line}" )
                    
                    sanitized_snippet = "".join([ch if ord(ch) < 128 else ' ' for ch in snippet])
                    
                    category = "general"
                    if kind in {clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_TEMPLATE}:
                        category = "class"
                    elif kind in {clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.FUNCTION_TEMPLATE}:
                        category = "function"
                    elif kind == clang.cindex.CursorKind.ENUM_DECL:
                        category = "enum"
                    elif kind in {clang.cindex.CursorKind.CLASS_TEMPLATE, clang.cindex.CursorKind.FUNCTION_TEMPLATE}:
                        category = "template"

                    extracted_elements.append({
                        "name": name,
                        "category": category,
                        "file": current_file,
                        "line": node.location.line,
                        S_ITEM: sanitized_snippet
                    })

            for child in node.get_children():
                _traverse(child, current_file)

        _traverse(translation_unit.cursor, filepath)
        return extracted_elements

    def _m_get_node_source(
        self,
        node: clang.cindex.Cursor,
        filepath: str) -> str:
        try:
            extent = node.extent
            start = extent.start
            end = extent.end
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if start.line == end.line:
                return lines[start.line - 1][start.column - 1:end.column - 1]
            
            snippet_lines = []
            for l_idx in range(start.line - 1, end.line):
                if l_idx == start.line - 1:
                    snippet_lines.append(lines[l_idx][start.column - 1:])
                elif l_idx == end.line - 1:
                    snippet_lines.append(lines[l_idx][:end.column - 1])
                else:
                    snippet_lines.append(lines[l_idx])
            return "".join(snippet_lines)
        except Exception:
            return ""


def scanDirectory(
    root_dir: str,
    analyzer: CParser) -> Dict[str, List[Dict[str, Any]]]:

    extensions = {'.cpp', '.hpp', '.h', '.cc', '.cxx'}
    datasets = {"class": [], "function": [], "enum": [], "template": [], "general": []}
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                logging.info(f"Parsing {filepath}")
                elements = analyzer.parse_file(filepath)
                for elem in elements:
                    cat = elem["category"]
                    if cat in datasets:
                        datasets[cat].append(elem)
                    else:
                        datasets["general"].append(elem)
    return datasets


# P2 : Training/MLM
def trainEncoder(
    datasets: Dict[str, List[Dict[str, Any]]],
    model_name: str,
    mlm_prob: float,
    seed: int):

    random.seed(seed)
    torch.manual_seed(seed)
    
    logging.info(f"Init tokenizer {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    all_snippets = []
    for cat, items in datasets.items():
        for item in items:
            if item[S_ITEM].strip():
                all_snippets.append(item[S_ITEM])

    if not all_snippets:
        logging.warning("No snippets, using dummy text for completeness.")
        all_snippets = ["int main() { return 314159; }"]

    tokenized_data = tokenizer(all_snippets, truncation=True, padding=True, max_length=512, return_tensors="pt")

    class InternalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings.input_ids)
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    train_dataset = InternalDataset(tokenized_data)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob / 100.0
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        seed=seed,
        logging_steps=5,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logging.info("Fine-tuning...")
    trainer.train()

    return model, tokenizer



# P3/4: Perplexity & Anomaly Detection
def computePerplexity(
    text: str,
    model, tokenizer,
    max_length: int = 512,
    overlap: float = 0.1) -> float:

    device = model.device
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"][0]
    
    seq_len = input_ids.size(0)
    if seq_len == 0:
        return 0.0

    stride = int(max_length * (1 - overlap))
    nlls = []

    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        chunk_ids = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        
        target_ids = chunk_ids.clone()

        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * chunk_ids.size(1)
            nlls.append(neg_log_likelihood)

        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / seq_len).item()
    return ppl

def calcThreshold(
    model,
    tokenizer,
    datasets):
    
    threshold = 100.0
    
    validation_ppls = []
    for cat, items in datasets.items():
        for item in items:
            if item[S_ITEM].strip():
                ppl = computePerplexity(item[S_ITEM], model, tokenizer)
                validation_ppls.append(ppl)
    if validation_ppls:
        mean_m = sum(validation_ppls) / len(validation_ppls)
        std_d = math.sqrt(sum((x - mean_m) ** 2 for x in validation_ppls) / len(validation_ppls))
        threshold = mean_m + (2 * std_d)
        logging.info(f"Threshold -> Mean (M): {mean_m:.2f}, Std (D): {std_d:.2f}")
    else:
        logging.warning("No snippets")

    return threshold

def runAnalyze(
    root_dir: str,
    model,
    tokenizer,
    threshold: float):

    extensions = {

        '.h',
        '.hpp',
        '.hxx',
        '.c',
        '.cc',
        '.cpp',
        '.cxx'
    }
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_ppl = computePerplexity(content, model, tokenizer)
                    logging.info(f"File: {filepath} | Perplexity: {file_ppl:.2f}")

                    if file_ppl > threshold:
                        logging.warning(f"[ANOMALY] File exceeds perplexity threshold ({file_ppl:.2f} > {threshold}): {filepath}")
                except Exception as e:
                    logging.error(f"Error analyzing file {filepath}: {e}")

# logger.cfg
def setupLogging(
    log_file: str = None,
    log_level: str = "INFO"):

    h = logging.StreamHandler()
    if log_file:
        h = logging.FileHandler(log_file)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    h.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(h)

def saveModel(
    dirModel: str,
    model,
    tokenizer,
    threshold):

    model.save_pretrained(dirModel)
    tokenizer.save_pretrained(dirModel)
    logging.info(f"Model saved {dirModel}")

    with open(os.path.join(dirModel, F_THRESHOLD), "w") as f:
        f.write(str(threshold))

    return

def loadModel(
    dirModel: str):

    loaded_tokenizer = AutoTokenizer.from_pretrained(dirModel)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(dirModel)
    logging.info(f"Model loaded {dirModel}")
    
    threshold = 100.0
    with open(os.path.join(dirModel, F_THRESHOLD), "r") as f:
        threshold = float(f.read())

    return loaded_model, loaded_tokenizer, threshold

# entry point
def main():

    parser = argparse.ArgumentParser(description="C++ Anomaly Detection")
    parser.add_argument("--model", required=True, help="Name of the Encoder model")
    parser.add_argument("--dir", required=True, help="Directory containing the C++ (.h/.cpp) files to be processed")
    parser.add_argument("--std", default="c++17", help="C/C++ language standard (e.g. 'c11', 'c++17')")
    parser.add_argument("--mlm", type=float, default=15.0, help="Masked Language Modeling probability percentage (Default: 15)")
    parser.add_argument("--model-in", default="", help="Path to the trained model folder")
    parser.add_argument("--model-out", default="", help="Path to the folder where the trained model will be saved")
    parser.add_argument("--log-file", help="Path to the log file")
    parser.add_argument("--log-level", default="INFO", help="Log level (INFO, DEBUG, WARNING, ERROR)")
    args = parser.parse_args()

    setupLogging(args.log_file, args.log_level)

    if len(args.model_in) > 0:
        model, tokenizer, threshold = loadModel(args.model_in)
        logging.info("Phase_1/2/3 skipped")
    else:
        logging.info("Phase_1: Parsing ...")
        analyzer = CParser(std_flag=args.std)
        datasets = scanDirectory(args.dir, analyzer)
        logging.info("Phase_2: Training ...")
        model, tokenizer = trainEncoder(datasets, args.model, args.mlm, seed=42)
        logging.info("Phase_3: Pseudo-Perplexity ...")
        threshold = calcThreshold( model, tokenizer, datasets)
    
        model.save_pretrained(args.model_out)
        tokenizer.save_pretrained(args.model_out)
        logging.info(f"Model saved {args.model_out}")

    logging.info("Phase_4: Reporting ...")
    logging.info(f"Threshold (M + 2D): {threshold:.2f}")
    runAnalyze(args.dir, model, tokenizer, threshold)

    if len(args.model_out) > 0:
        saveModel(args.model_out, model, tokenizer, threshold)

if __name__ == "__main__":

    main()
