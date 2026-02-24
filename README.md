# Sentiment Analysis of Climate Change Denialism on Twitter

Análisis de sentimiento aplicado a tweets en inglés sobre negacionismo climático. Utiliza el modelo **CardiffNLP Twitter-RoBERTa** para clasificar cada tweet como positivo (POS), negativo (NEG) o neutro (NEU). Jose Carlos Monescillo.

---

## Instalación

```bash
git clone <url-del-repositorio>
cd climate_sentiment

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

> La primera ejecución descarga el modelo (~500 MB) desde Hugging Face.

---

## Preparar los datos

El repositorio incluye `train.csv` y `test.csv`. Para generar el archivo combinado:

```bash
python scripts/combine_data.py
```

Esto crea `data/tweets_combinado.csv` a partir de los datos originales.

---

## Uso

```bash
python main.py --input data/tweets_combinado.csv
```

### Argumentos

| Argumento    | Descripción                 |
|-------------|-----------------------------|
| `--input`   | Ruta al CSV de entrada     |
| `--no-charts` | Omitir generación de gráficos |

---

## Formato de entrada

El CSV debe tener al menos la columna `text`. El script `combine_data.py` genera un archivo con `text`, `date` y `evento`.

---

## Salida

Los resultados se guardan en `results/`:

```
results/
├── denial_sentiment.csv          # tweets clasificados
├── denial_sentiment_stats.json   # estadísticas
└── charts/
    ├── sentiment_bars.png
    ├── sentiment_pie.png
    ├── confidence_histogram.png
    ├── wordcloud.png
    ├── top_keywords.png
    └── top_examples.txt
```

---

## Estructura del proyecto

```
climate_sentiment/
├── data/
│   ├── train.csv              # dataset original (train)
│   ├── test.csv               # dataset original (test)
│   └── tweets_combinado.csv   # generado por combine_data.py
├── scripts/
│   └── combine_data.py        # combina train + test
├── config.py                  # palabras clave de negacionismo
├── main.py                    # pipeline principal
├── utils.py                   # utilidades
├── requirements.txt
└── README.md
```

### Archivos que no se suben a Git

Según `.gitignore`: `venv/`, `results/`, `data/tweets_combinado.csv`, caché de modelos.

---

## Tecnologías

- [CardiffNLP Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- pandas, matplotlib, wordcloud, tqdm

---

## Autor

**Jose Carlos Monescillo** — TFM
