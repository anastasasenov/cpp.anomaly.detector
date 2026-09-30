## C/C++ Anomaly Detection Pipeline

This experimental Python pipeline parses C/C++ source files (.h, .c, .cpp) using clang.cindex, trains a transformer model and detects code anomalies. Think of it as a digital bloodhound for code smells: it captures that exact moment when you stare at a pull request, sense that something is deeply wrong, but your brain refuses to formulate why in a proper sentence. Our model detects the anomalies your intuition smelled first.

### How It Works Under the Hood
    * Phase 1: Code Parsing
    * Phase 2: Model Training       
    * Phase 3: Thresholding / Perplexity Calculation
    * Phase 4: Deviation Analysis & Reporting

#### Phase 1: AST Parsing & ASCII Sanitization

Recursively scans target directories for C++ source and header files (.cpp, .hpp, .h, .cc, .cxx).

Utilizes Clang's AST parser to separate classes, functions, methods, structs, templates, and enums alongside their code snippets.

Automatically filters out comments and non-ASCII characters, reporting exact file and line numbers for any non-ASCII identifiers found.

#### Phase 2: Model Training (Encoder MLM Fine-Tuning)

Initializes a RoBERTa-style or BERT-style Encoder architecture (AutoModelForMaskedLM).

Fine-tunes the model on segmented code datasets using Masked Language Modeling with reproducible seeding.

#### Phase 3: Dynamic Thresholding & Perplexity Calculation

Computes Pseudo-Perplexity (PPL) using cross-entropy loss across tokens.

Applies a sliding window with a 10% overlap for files exceeding the model's token limit.

Establishes a dynamic statistical threshold (median + 2 * deviation) across validation splits to flag anomalies.

#### Phase 4: Deviation Analysis & Reporting

Scans target source files against the trained model to detect anti-patterns, poor syntax, or hidden logic bugs with exact file and line number reporting.

### Prerequisites 

    $ pip install clang transformers torch datasets

### Command Line Arguments

The script utilizes argparse for flexible CLI configuration:

    --model     The name of Encoder model.
    --dir       The path to the directory containing the .h and .cpp files to be processed.
    --std       ( optional ) C/C++ language standard (e.g. 'c11', 'c++17')
    --mlm       ( optional ) Masked Language Modeling probability percentage (Default: 15)
    --log-file  ( optional ) Path to the log file.
    --log-level ( optional ) Log level (INFO, DEBUG, WARNING, ERROR).
   
### Usage

    $ ./cpp.anomaly.detector.py --model codebert --dir ./src --std c99 --mlm 15 --log-file log.txt --log-level INFO
    
### License

Distributed under the MIT License.
