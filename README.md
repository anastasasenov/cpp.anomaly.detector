## Clang-based Documentation Generator (C++ Code-to-Comment Pipeline)

This experimental Python pipeline parses C/C++ source files (.h, .c, .cpp) using clang.cindex, trains a transformer model and detects code anomalies.

### How It Works Under the Hood

    * Phase 1: Code Parsing

    * Phase 2: Model Training
        
    * Phase 3: Thresholding / Perplexity Calculation

    * Phase 4: Deviation Analysis & Reporting
    
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
