# Models environment setup

## Installation steps
1. Enter the terminal of either `Encoder`, `Synthesyzer` or `vocoder`.
2. ```bash
    conda create --prefix ./.env
   ```
3. ```bash
   conda activate ./.env
    ```
4.  ```bash
    conda env update --prefix ./.env --file <your local yml requirements file here> --prune
    ```

If you want to disable that environment, then on the same
folder path, enter in terminal:
```bash
    conda deactivate
```