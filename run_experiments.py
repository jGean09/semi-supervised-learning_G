"""
run_experiments.py

Roda o pipeline geral (seleciona o melhor classificador -> tenta rotular todas as
instâncias não rotuladas -> quando nenhuma das instâncias atingem o nível de
confiança definido, aplica o índice Silhouette) em todos os datasets da pasta
`datasets/` e salva os resultados (acurácia, tempo de execução, iterações,
critério de parada etc.) em um arquivo CSV.

As duas variantes disponíveis na reavaliação de instâncias fracas são:

1. MySelfNewEssembleCPCommittee:
    Após o Silhouette, caso sejam encontradas instâncias com índice inferior
    ao threshold, seus pseudo-rótulos são removidos (-1) e elas são
    imediatamente reavaliadas pelo comitê de classificadores.

2. MySelfNewEssembleCP:
    Após o Silhouette, caso sejam encontradas instâncias com índice inferior
    ao threshold, seus pseudo-rótulos são removidos (-1) e elas são
    reavaliadas pelo próprio especialista na próxima iteração.


Como usar:
    Coloque este arquivo na raiz do projeto (mesmo nível da pasta 'datasets/'
    e dos arquivos reevaluation_of_labels.py, selfNewEssembleCP.py, etc.) e rode:

        python run_experiments.py

    A cada execução, um arquivo CSV NOVO é criado em `results/` (nunca
    sobrescreve um anterior), com timestamp no nome. O script pergunta se
    você quer dar um nome pra essa rodada -- aperte Enter pra pular e usar
    só o timestamp. Se preferir não ser perguntado (ex: rodando em lote),
    passe o nome direto:

        python run_experiments.py --name car_sem_comite

    Dentro de cada rodada, os resultados vão sendo salvos incrementalmente
    -- se algum dataset travar ou você precisar interromper (Ctrl+C), o que
    já rodou naquele arquivo fica salvo.

Observação sobre desempenho:
    O cálculo de silhouette (`silhouette_samples`) é O(n^2) em relação ao
    número de instâncias rotuladas, e ele é recalculado a cada iteração em
    que nenhuma amostra bate o threshold. Em datasets grandes (Madelon,
    Mushroom, KrVsKp, Semeion, Twonorm, Waveform, MultipleFeaturesKarhunen,
    ImageSegmentation...) isso pode deixar a execução bem mais lenta,
    especialmente se o algoritmo entrar em oscilação (não convergir antes
    do max_iter). Ajuste MAX_ITER abaixo se precisar limitar isso.
"""

import argparse
import csv
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from random import choice

from numpy import where
from pandas import read_csv
from selfNewEssemble import MySelfNewEssemble
from selfNewEssembleCP import MySelfNewEssembleCP
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import select_labels

##################################################
# Configurações do experimento

DATASETS_DIR = "datasets"
OUTPUT_DIR = "results"

# Parâmetros do MySelfNewEssembleCP usados em todos os datasets
THRESHOLD = 0.75
SILHOUETTE_THRESHOLD = -0.2
MAX_ITER = 100
VERBOSE = False  # deixe True se quiser acompanhar o log de cada dataset

# Se quiser rodar só alguns datasets específicos, liste os nomes de arquivo
# aqui (ex: ["Car.csv", "GermanCredit.csv"]). Deixe None para rodar tudo
# que estiver em DATASETS_DIR.
DATASETS_WHITELIST = None

CLASSIFIERS = {
    "KNN": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "Logistic Regression": LogisticRegression,
    "Neural Network": MLPClassifier,
}

FIELDNAMES = [
    "dataset",
    "modo",
    "n_amostras_total",
    "n_atributos",
    "n_treino_inicial",
    "melhor_modelo",
    "acuracia_antes_self_training",
    "acuracia_apos_self_training",
    "criterio_parada",
    "n_iteracoes",
    "instancias_nao_rotuladas_antes",
    "instancias_nao_rotuladas_depois",
    "tempo_selecao_modelo_s",
    "tempo_comite_s",
    "tempo_self_training_s",
    "tempo_total_s",
    "erro",
]


def build_model(model_name, model_cls):
    if model_name == "Logistic Regression":
        return model_cls(max_iter=1000)
    elif model_name == "Neural Network":
        return model_cls(max_iter=3000)
    return model_cls()


def run_pipeline_for_dataset(csv_path: Path, mode: str) -> dict:
    dataset_name = csv_path.name
    row = {field: "" for field in FIELDNAMES}
    row["dataset"] = dataset_name
    row["modo"] = mode

    t_start_total = time.perf_counter()

    try:
        df = read_csv(csv_path, header=0)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        row["n_amostras_total"] = X.shape[0]
        row["n_atributos"] = X.shape[1]

        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        X_train, _, y_train, _ = train_test_split(
            X_train_all,
            y_train_all,
            test_size=0.95,
            random_state=42,
            stratify=y_train_all,
        )
        row["n_treino_inicial"] = len(X_train)

        ##########################################
        # Seleção do melhor classificador
        t0 = time.perf_counter()
        results = {name: 0.0 for name in CLASSIFIERS.keys()}
        for name, model_cls in CLASSIFIERS.items():
            model = build_model(name, model_cls)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_all)
            results[name] = accuracy_score(y_test_all, y_pred)

        max_acc = max(results.values())
        best_models = [name for name, acc in results.items() if acc == max_acc]
        best_model_name = choice(best_models)
        best_model_cls = CLASSIFIERS[best_model_name]
        row["melhor_modelo"] = best_model_name
        row["acuracia_antes_self_training"] = round(max_acc, 4)
        row["tempo_selecao_modelo_s"] = round(time.perf_counter() - t0, 3)

        ##########################################
        # Comitê (sempre montado para registrar o tempo;
        # só é passado ao especialista no modo 'com_comite').
        t0 = time.perf_counter()
        models = [(name, build_model(name, cls)) for name, cls in CLASSIFIERS.items()]
        weights = []
        for name, model in models:
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_train)
            weights.append(accuracy_score(y_train, y_pred))
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        committee = VotingClassifier(
            estimators=models, voting="soft", weights=normalized_weights, verbose=False
        )
        committee.fit(X_train, y_train)
        row["tempo_comite_s"] = round(time.perf_counter() - t0, 3)

        ##########################################
        # Self-training (especialista) — modo selecionado via --mode
        if mode == "com_comite":
            specialist = MySelfNewEssemble(
                base_estimator=build_model(best_model_name, best_model_cls),
                committee=committee,
                threshold=THRESHOLD,
                max_iter=MAX_ITER,
                silhouette_threshold=SILHOUETTE_THRESHOLD,
                verbose=VERBOSE,
            )
        else:  # sem_comite
            specialist = MySelfNewEssembleCP(
                base_estimator=build_model(best_model_name, best_model_cls),
                threshold=THRESHOLD,
                max_iter=MAX_ITER,
                silhouette_threshold=SILHOUETTE_THRESHOLD,
                verbose=VERBOSE,
            )

        y_2 = select_labels(y_train_all, X_train_all, 0.05)
        num_unlabeled_before = int(sum(where(y_2 == -1, 1, 0)))
        row["instancias_nao_rotuladas_antes"] = num_unlabeled_before

        t0 = time.perf_counter()
        specialist.fit(X_train_all, y_2)
        row["tempo_self_training_s"] = round(time.perf_counter() - t0, 3)

        y_pseudo = specialist.transduction_
        num_unlabeled_after = int(sum(where(y_pseudo == -1, 1, 0)))
        row["instancias_nao_rotuladas_depois"] = num_unlabeled_after

        y_pred = specialist.predict(X_test_all)
        accuracy = accuracy_score(y_test_all, y_pred)
        row["acuracia_apos_self_training"] = round(accuracy, 4)
        row["criterio_parada"] = specialist.termination_condition_
        row["n_iteracoes"] = specialist.n_iter_

    except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer erro por dataset
        row["erro"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERRO] {dataset_name}: {exc}")
        traceback.print_exc()

    row["tempo_total_s"] = round(time.perf_counter() - t_start_total, 3)
    return row


def discover_datasets() -> list[Path]:
    all_csvs = sorted(Path(DATASETS_DIR).glob("*.csv"))
    if DATASETS_WHITELIST:
        all_csvs = [p for p in all_csvs if p.name in DATASETS_WHITELIST]
    return all_csvs


def slugify(nome: str) -> str:
    """Deixa o nome seguro pra usar em nome de arquivo (sem espaço/acentos/símbolos)."""
    nome = nome.strip().lower()
    nome = re.sub(r"\s+", "_", nome)
    nome = re.sub(r"[^a-z0-9_\-]", "", nome)
    return nome


def build_output_path(nome_rodada: str | None) -> Path:
    """Monta um caminho de CSV novo a cada chamada (timestamp garante que
    nunca sobrescreve um arquivo de uma rodada anterior)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(nome_rodada) if nome_rodada else ""
    filename = f"experiment_results_{timestamp}" + (f"_{slug}" if slug else "") + ".csv"

    output_path = Path(OUTPUT_DIR) / filename
    # Garantia extra: se por algum motivo já existir (duas rodadas no mesmo
    # segundo), acrescenta um sufixo numérico em vez de sobrescrever.
    contador = 2
    base_path = output_path
    while output_path.exists():
        output_path = base_path.with_name(f"{base_path.stem}_{contador}{base_path.suffix}")
        contador += 1
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Roda o experimento de self-training em todos os datasets.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome opcional pra essa rodada (vira parte do nome do arquivo CSV). "
        "Se não passar, o script pergunta interativamente.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["com_comite", "sem_comite"],
        default=None,
        help="Algoritmo a usar: 'com_comite' (MySelfNewEssemble) ou "
        "'sem_comite' (MySelfNewEssembleCP). "
        "Se não passar, o script pergunta interativamente.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    nome_rodada = args.name
    if nome_rodada is None:
        try:
            nome_rodada = input(
                "Nome para essa rodada (opcional, ex: 'sem_comite_car'; Enter pra pular): "
            ).strip()
        except EOFError:
            nome_rodada = ""

    mode = args.mode
    if mode is None:
        try:
            escolha = input(
                "Modo do algoritmo — [1] com_comite (MySelfNewEssemble) "
                "ou [2] sem_comite (MySelfNewEssembleCP) [padrão: 2]: "
            ).strip()
        except EOFError:
            escolha = ""
        mode = "com_comite" if escolha == "1" else "sem_comite"

    # Inclui o modo no nome do arquivo automaticamente se não estiver já lá
    if mode not in (nome_rodada or ""):
        nome_rodada = f"{nome_rodada}_{mode}" if nome_rodada else mode

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    output_path = build_output_path(nome_rodada)

    datasets = discover_datasets()
    if not datasets:
        print(f"Nenhum CSV encontrado em '{DATASETS_DIR}/'. Confira o caminho.")
        return

    print(f"\nModo: {mode}")
    print(f"Encontrados {len(datasets)} datasets. Resultados desta rodada serão salvos em: {output_path}\n")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, csv_path in enumerate(datasets, start=1):
            print(f"[{i}/{len(datasets)}] Rodando: {csv_path.name} ...")
            row = run_pipeline_for_dataset(csv_path, mode)
            writer.writerow(row)
            f.flush()

            status = "OK" if not row["erro"] else f"ERRO ({row['erro']})"
            print(
                f"    -> {status} | modo={mode} | especialista={row['melhor_modelo']} "
                f"| acurácia={row['acuracia_apos_self_training']} "
                f"| iterações={row['n_iteracoes']} | parada={row['criterio_parada']} "
                f"| tempo_total={row['tempo_total_s']}s\n"
            )

    print(f"\nConcluído. Resultados em: {output_path}")


if __name__ == "__main__":
    main()