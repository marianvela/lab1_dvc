# src/preprocess.py
import pandas as pd
import sys

def preprocess(input_file, output_file, features, target):
    # cargar dataset
    df = pd.read_csv(input_file)

    # eliminar filas con nulos
    df = df.dropna()

    # seleccionar columnas
    columns = features + [target]
    df = df[columns]

    # guardar dataset limpio
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Data saved to {output_file}")

if __name__ == "__main__":
    # argumentos: input , output , params
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    # leer parametros del yaml
    import yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)