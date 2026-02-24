# Sentiment Analysis of Climate Change Denialism on Twitter

Análisis de sentimiento aplicado a tweets en inglés sobre negacionismo climático. Utiliza el modelo **CardiffNLP Twitter-RoBERTa** para clasificar cada tweet como positivo (POS), negativo (NEG) o neutro (NEU). Proyecto del TFM de Jose Carlos Monescillo.

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

## Datos

El análisis requiere un CSV con tweets. Opciones:

1. **CSV propio**: coloca tu archivo en `data/` con al menos la columna `text`.

2. **Si tienes train.csv y test.csv**: ejecuta el script para combinarlos:
   ```bash
   python scripts/combine_data.py
   ```
   Esto genera `data/tweets_combinado.csv` (necesario que existan `data/train.csv` y `data/test.csv` con columna `message`).

El archivo `tweets_combinado.csv` no se sube a Git; cada usuario debe generarlo o usar su propio dataset.

---

## Uso

```bash
python main.py --input data/tweets_combinado.csv
```

### Argumentos

| Argumento     | Descripción                    |
|---------------|--------------------------------|
| `--input`     | Ruta al CSV de entrada         |
| `--no-charts` | Omitir generación de gráficos  |

---

## Formato de entrada

El CSV debe tener al menos la columna `text`:

```csv
text
"Climate change is a hoax invented by the Chinese"
"I can't believe people still deny global warming"
```

---

## Salida

Los resultados se guardan en `results/`:

```
results/
├── denial_sentiment.csv          # tweets clasificados (text, sentiment, score)
├── denial_sentiment_stats.json   # estadísticas
└── charts/
    ├── sentiment_bars.png       # distribución POS/NEU/NEG
    ├── sentiment_pie.png        # proporciones
    ├── confidence_histogram.png  # confianza del modelo
    ├── wordcloud.png
    ├── top_keywords.png         # palabras más frecuentes
    └── top_examples.txt         # ejemplos por sentimiento
```

---

## Estructura del proyecto

```
climate_sentiment/
├── data/
│   └── tweets_combinado.csv   # CSV de entrada (no en Git; añadir o generar)
├── scripts/
│   └── combine_data.py        # opcional: combina train.csv + test.csv
├── results/                   # resultados generados
│   ├── denial_sentiment.csv
│   ├── denial_sentiment_stats.json
│   └── charts/
├── config.py                  # palabras clave de negacionismo
├── main.py                    # pipeline principal
├── utils.py                   # carga, filtrado y exportación
├── requirements.txt
├── .gitignore
└── README.md
```

### Excluidos en .gitignore

- `venv/` — entorno virtual
- `data/tweets_combinado.csv` — datos generados
- `__pycache__/`, `.cache/` — caché de Python y modelos

---

## Tecnologías

- [CardiffNLP Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- pandas, matplotlib, wordcloud, tqdm

---

## Autor

**Jose Carlos Monescillo** — TFM
