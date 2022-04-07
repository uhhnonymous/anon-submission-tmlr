# condo-adapter

ConDo Adapter performs Confounded Domain Adaptation, which corrects for
batch effects while conditioning on confounding variables.
We hope it sparks joy as you clean up your data!

## Installation

### Installation from source

After cloning this repo, install the dependencies on the command-line via:

```console
pip install -r requirements.txt
```

In this directory, run

```console
pip install -e .
```

## Usage

Import ConDo and create the adapter:
```python
import condo
condoer = condo.ConDoAdapter()
```

Try using it:
```python
import numpy as np
X_T = np.sort(np.random.uniform(0, 8, size=(100, 1)))
X_S = np.sort(np.random.uniform(4, 8, size=(100, 1)))
Y_T = np.random.normal(4 * X_T + 1, 1 * X_T + 1)
Y_Strue = np.random.normal(4 * X_S + 1, 1 * X_S + 1)
Y_S = 5 * Y_Strue + 2
condoer.fit(Y_S, Y_T, X_S, X_T)
Y_S2T = condoer.transform(Y_S)
print(f"before ConDo: {np.mean((Y_S - Y_Strue) ** 2):.3f}")
print(f"after ConDo:  {np.mean((Y_S2T - Y_Strue) ** 2):.3f}")
```

More thorough examples are provided in the examples directory.

## Development

### Testing
In this directory run
```console
pytest
```

### Code formatting
The Uncompromising Code Formatter: [Black](https://github.com/psf/black)  
```black {source_file_or_directory}```  

Install it into pre-commit hook to always commit well-formatted code:  
```pre-commit install```
