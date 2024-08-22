import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os

# Function to run a notebook and save it as a Python script
def run_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(notebook, {'metadata': {'path': './'}})

    # Convert notebook to Python script
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(notebook)

    # Save the script
    script_path = notebook_path.replace(".ipynb", ".py")
    with open(script_path, 'w') as f:
        f.write(script)

    # Optionally, execute the Python script directly
    os.system(f'python {script_path}')

# Run the notebooks sequentially
run_notebook('HandlingMissingValue.ipynb')
run_notebook('HandlingOutliar.ipynb')
run_notebook('Model.ipynb')