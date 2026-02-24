"""
Combina train.csv y test.csv en tweets_combinado.csv.
Ejecutar antes del análisis si no existe el archivo combinado.

    python scripts/combine_data.py
"""
import os
import pandas as pd

DATA_DIR = "data"
TRAIN = os.path.join(DATA_DIR, "train.csv")
TEST = os.path.join(DATA_DIR, "test.csv")
OUT = os.path.join(DATA_DIR, "tweets_combinado.csv")


def main():
    if not os.path.exists(TRAIN) or not os.path.exists(TEST):
        print(f"  Error: Necesitas train.csv y test.csv en {DATA_DIR}/")
        return 1

    df_train = pd.read_csv(TRAIN)
    df_test = pd.read_csv(TEST)

    # train: sentiment, message, tweetid
    # test: message, tweetid
    col = "message" if "message" in df_train.columns else "text"
    if col not in df_train.columns:
        print(f"  Error: Columnas en train: {list(df_train.columns)}")
        return 1

    df_train = df_train[[col]].rename(columns={col: "text"})
    df_test = df_test[[col]].rename(columns={col: "text"})

    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined["date"] = ""
    combined["evento"] = ""
    combined = combined[["text", "date", "evento"]]

    os.makedirs(DATA_DIR, exist_ok=True)
    combined.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"  Creado: {OUT} ({len(combined)} tweets)")
    return 0


if __name__ == "__main__":
    exit(main())
