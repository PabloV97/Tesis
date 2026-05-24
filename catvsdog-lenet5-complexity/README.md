# Cat vs Dog LeNet5 Complexity

Pipeline para entrenar una red LeNet5 en Cats vs Dogs, guardar activaciones por neurona y calcular medidas de complejidad con correlacion de Spearman contra accuracy/loss del modelo.

## Ejecucion recomendada en Jupyter

Abrir:

```text
run_catvsdog_lenet5_complexity.ipynb
```

Ejecutar primero la prueba corta:

```python
%run "$SCRIPT" --epochs 1,2 --target-layer fc2 --n-samples 200 --local-data-dir "$LOCAL_DATA_DIR" --out-dir "$OUT_DIR"
```

Cuando funcione, ejecutar la corrida completa:

```python
%run "$SCRIPT" --epochs 10,20,30,40,50 --target-layer fc2 --n-samples 1000 --local-data-dir "$LOCAL_DATA_DIR" --out-dir "$OUT_DIR"
```

## Formatos de dataset aceptados

El argumento `--local-data-dir` puede apuntar a una carpeta con cualquiera de estas estructuras:

```text
dataset/
  PetImages/
    Cat/
    Dog/
```

```text
dataset/
  cat.1.jpg
  dog.1.jpg
```

```text
dataset/
  train/
    cats/
    dogs/
  validation/
    cats/
    dogs/
```

## Salidas

Por defecto se guardan dentro de `catvsdog_lenet5_outputs/`:

- `activations_by_epoch/`: activaciones completas por epoca.
- `datasets_conv1_fc2/`: datasets de activaciones entre `conv1` y `fc2`.
- `results/metrics_summary.csv`: accuracy/loss por epoca.
- `results/complexity/`: medidas de complejidad.
- `results/correlations/`: correlaciones Spearman.
