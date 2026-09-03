"""
run_exp.py

Roda o experimento de self-training (MySelfNewEssembleCP) em datasets
binarios, com StratifiedKFold, multiplos percentuais e seeds.

Baseado em: reevaluation_of_labels.py + selfNewEssembleCP.py

Como usar
---------
  # Ver todos os datasets disponíveis:
      python run_exp.py --list

  # Rodar em um único dataset:
      python run_exp.py --datasets Haberman

  # Rodar em vários datasets:
      python run_exp.py --datasets Haberman GermanCredit Pima

  # Rodar em TODOS os datasets binários:
      python run_exp.py

  # Personalizar parâmetros:
      python run_exp.py --datasets Haberman --pct 0.05 0.10 --seeds 42 --folds 5
"""

from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path
from random import choice

import numpy as np
import pandas as pd
from selfNewEssembleCP import MySelfNewEssembleCP
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import select_labels

# ==============================================================
# Configurações (mesmas de reevaluation_of_labels.py)
# ==============================================================

DATASETS_DIR = Path("datasets")
OUTPUT_DIR   = Path("results")

THRESHOLD            = 0.75   # confiança mínima para aceitar um pseudo-rótulo
SILHOUETTE_THRESHOLD = -0.2   # limite do índice Silhouette para descartar instâncias fracas
MAX_ITER             = 100    # máximo de iterações do self-training

# Classificadores candidatos a especialista (igual a reevaluation_of_labels.py)
CLASSIFIERS = {
    "KNN":                 KNeighborsClassifier,
    "Naive Bayes":         GaussianNB,
    "Decision Tree":       DecisionTreeClassifier,
    "Random Forest":       RandomForestClassifier,
    "XGBoost":             XGBClassifier,
    "Logistic Regression": LogisticRegression,
    "Neural Network":      MLPClassifier,
}


# ==============================================================
# Funções auxiliares (mesma lógica de reevaluation_of_labels.py)
# ==============================================================

def instanciar_classificador(nome, cls):
    """Instancia o classificador com parâmetros padrão do projeto."""
    if nome == "Logistic Regression":
        return cls(max_iter=1000)
    if nome == "Neural Network":
        return cls(max_iter=3000)
    return cls()


def escolher_especialista(X_train, y_train, X_test, y_test):
    """
    Avalia cada classificador nos dados rotulados e devolve
    o nome e a classe do melhor (critério: acurácia no teste).
    Lógica igual a reevaluation_of_labels.py.
    """
    resultados = {}
    for nome, cls in CLASSIFIERS.items():
        modelo = instanciar_classificador(nome, cls)
        modelo.fit(X_train, y_train)
        acc = accuracy_score(y_test, modelo.predict(X_test))
        resultados[nome] = acc

    # Pega a maior acurácia e, em caso de empate, escolhe aleatoriamente
    max_acc    = max(resultados.values())
    candidatos = [n for n, a in resultados.items() if a == max_acc]
    melhor     = choice(candidatos)
    return melhor, CLASSIFIERS[melhor]


def normalizar_rotulos(y):
    """Converte rótulos quaisquer para 0/1 (necessário para select_labels)."""
    classes = np.unique(y)
    mapa    = {c: i for i, c in enumerate(classes)}
    return np.array([mapa[v] for v in y], dtype=int)


# ==============================================================
# Loop principal de avaliação
# ==============================================================

def avaliar_dataset(nome_dataset, X, y, percentuais, seeds, n_folds, out_file, first_write):
    """
    Roda o experimento para um dataset e salva os resultados no CSV.
    Mostra progresso detalhado para cada fold.
    """
    for pct in percentuais:
        for seed in seeds:

            print(f"\n  [pct={pct:.0%}  seed={seed}]")

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

                X_treino_tudo = X[train_idx]
                y_treino_tudo = y[train_idx]
                X_teste       = X[test_idx]
                y_teste       = y[test_idx]

                # -------------------------------------------------
                # Seleciona o subconjunto rotulado para escolher
                # o especialista (mesma lógica de reevaluation_of_labels.py)
                # -------------------------------------------------
                rng     = np.random.default_rng(seed + fold)
                indices = []
                for classe in np.unique(y_treino_tudo):
                    idx_classe = np.where(y_treino_tudo == classe)[0]
                    n_sel      = max(1, int(len(idx_classe) * pct))
                    indices.extend(
                        rng.choice(idx_classe, size=min(n_sel, len(idx_classe)), replace=False)
                    )

                X_rotulado = X_treino_tudo[indices]
                y_rotulado = y_treino_tudo[indices]

                # -------------------------------------------------
                # Escolhe o melhor classificador como especialista
                # -------------------------------------------------
                melhor_nome, melhor_cls = escolher_especialista(
                    X_rotulado, y_rotulado, X_teste, y_teste
                )

                # -------------------------------------------------
                # Aplica select_labels: mantém pct% rotulado,
                # o restante vira -1 (não rotulado)
                # -------------------------------------------------
                np.random.seed(seed + fold)  # garante reprodutibilidade
                y_semi = select_labels(y_treino_tudo.copy(), X_treino_tudo, pct)

                nao_rotulados_antes = int(np.sum(y_semi == -1))

                # -------------------------------------------------
                # Roda o self-training
                # -------------------------------------------------
                t0 = time.time()
                try:
                    especialista = MySelfNewEssembleCP(
                        base_estimator=instanciar_classificador(melhor_nome, melhor_cls),
                        threshold=THRESHOLD,
                        max_iter=MAX_ITER,
                        silhouette_threshold=SILHOUETTE_THRESHOLD,
                        verbose=False,
                    )
                    especialista.fit(X_treino_tudo, y_semi)

                    y_pred = especialista.predict(X_teste)
                    proba  = especialista.predict_proba(X_teste)

                    acc = accuracy_score(y_teste, y_pred)
                    f1  = f1_score(y_teste, y_pred, average="binary")
                    auc = roc_auc_score(y_teste, proba[:, 1])

                    nao_rotulados_depois = int(np.sum(especialista.transduction_ == -1))
                    pct_rotuladas        = (len(y_semi) - nao_rotulados_depois) / len(y_semi)

                    tempo = round(time.time() - t0, 3)

                    # Mostra progresso do fold
                    print(
                        f"    Fold {fold:>2}/{n_folds}  "
                        f"especialista={melhor_nome:<20}  "
                        f"acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
                        f"rotuladas={pct_rotuladas:.1%}  "
                        f"iter={especialista.n_iter_:>3}  "
                        f"parada={especialista.termination_condition_}  "
                        f"({tempo}s)"
                    )

                    linha = {
                        "dataset":         nome_dataset,
                        "pct":             pct,
                        "seed":            seed,
                        "fold":            fold,
                        "especialista":    melhor_nome,
                        "nao_rot_antes":   nao_rotulados_antes,
                        "nao_rot_depois":  nao_rotulados_depois,
                        "acc":             round(acc, 6),
                        "f1":              round(f1, 6),
                        "auc":             round(auc, 6),
                        "pct_rotuladas":   round(pct_rotuladas, 6),
                        "n_iter":          especialista.n_iter_,
                        "criterio_parada": especialista.termination_condition_,
                        "tempo_fold_s":    tempo,
                        "erro":            None,
                    }

                except Exception as e:
                    tempo = round(time.time() - t0, 3)
                    print(f"    Fold {fold:>2}/{n_folds}  [ERRO] {e}")
                    traceback.print_exc()

                    linha = {
                        "dataset":         nome_dataset,
                        "pct":             pct,
                        "seed":            seed,
                        "fold":            fold,
                        "especialista":    melhor_nome,
                        "nao_rot_antes":   nao_rotulados_antes,
                        "nao_rot_depois":  None,
                        "acc":             None,
                        "f1":              None,
                        "auc":             None,
                        "pct_rotuladas":   None,
                        "n_iter":          None,
                        "criterio_parada": None,
                        "tempo_fold_s":    tempo,
                        "erro":            str(e),
                    }

                # Salva a linha imediatamente (incremental)
                pd.DataFrame([linha]).to_csv(
                    out_file, mode="a", header=first_write, index=False
                )
                first_write = False  # só escreve o cabeçalho na primeira linha

    return first_write


# ==============================================================
# CLI
# ==============================================================

def listar_datasets():
    """Imprime todos os CSVs disponíveis, indicando se são binários."""
    csvs = sorted(DATASETS_DIR.glob("*.csv"))
    if not csvs:
        print(f"Nenhum CSV encontrado em '{DATASETS_DIR}/'.")
        return
    print(f"\nDatasets em '{DATASETS_DIR}/' ({len(csvs)} total):\n")
    for p in csvs:
        try:
            y = pd.read_csv(p, header=0).iloc[:, -1].values
            n = len(np.unique(y))
            tag = "BINARIO" if n == 2 else f"{n} classes"
            print(f"  {p.name:<45} [{tag}]")
        except Exception as e:
            print(f"  {p.name:<45} [ERRO: {e}]")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Self-training (MySelfNewEssembleCP) em datasets binários."
    )
    parser.add_argument(
        "--datasets", nargs="*", metavar="NOME",
        help="Nome(s) do(s) dataset(s) (com ou sem .csv). Omita para rodar todos."
    )
    parser.add_argument(
        "--pct", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25],
        metavar="P", help="Percentuais de rótulos iniciais."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 7, 13, 21, 99],
        metavar="S", help="Seeds para reprodutibilidade."
    )
    parser.add_argument(
        "--folds", type=int, default=10,
        help="Número de folds no StratifiedKFold (padrão: 10)."
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Nome do arquivo CSV de saída (sem extensão)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Lista os datasets disponíveis e sai."
    )
    args = parser.parse_args()

    if args.list:
        listar_datasets()
        return

    # Define quais datasets processar
    if args.datasets:
        alvos = args.datasets
    else:
        alvos = [p.stem for p in sorted(DATASETS_DIR.glob("*.csv"))]

    # Arquivo de saída
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = args.name if args.name else f"resultados_{timestamp}"
    out_file     = OUTPUT_DIR / f"{nome_arquivo}.csv"
    first_write  = True  # controla se escreve o cabeçalho

    print(f"Percentuais : {args.pct}")
    print(f"Seeds       : {args.seeds}")
    print(f"Folds       : {args.folds}")
    print(f"Saída       : {out_file}")
    print(f"Datasets    : {alvos}\n")

    t_total = time.time()

    for i, ds_nome in enumerate(alvos, start=1):

        # Localiza o arquivo CSV
        path = DATASETS_DIR / (ds_nome if ds_nome.endswith(".csv") else f"{ds_nome}.csv")
        if not path.exists():
            print(f"[AVISO] Arquivo não encontrado: {path.name} — ignorando.\n")
            continue

        # Lê e prepara os dados
        df   = pd.read_csv(path, header=0)
        X    = df.iloc[:, :-1].values
        y    = normalizar_rotulos(df.iloc[:, -1].values)

        # Ignora datasets não binários
        if len(np.unique(y)) != 2:
            print(f"[IGNORADO] {path.name} ({len(np.unique(y))} classes — não binário)\n")
            continue

        print(f"{'='*60}")
        print(f"[{i}/{len(alvos)}] Dataset: {path.stem}  "
              f"({X.shape[0]} amostras, {X.shape[1]} atributos)")
        print(f"{'='*60}")

        t_ds = time.time()
        first_write = avaliar_dataset(
            nome_dataset=path.stem,
            X=X,
            y=y,
            percentuais=args.pct,
            seeds=args.seeds,
            n_folds=args.folds,
            out_file=out_file,
            first_write=first_write,
        )
        print(f"\n  Dataset concluído em {time.time() - t_ds:.1f}s\n")

    print(f"\nTudo concluído em {time.time() - t_total:.1f}s")
    print(f"Resultados em: {out_file}")


if __name__ == "__main__":
    main()
