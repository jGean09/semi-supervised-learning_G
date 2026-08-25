"""
plot_results.py

Lê um CSV de resultados gerado por run_experiments.py e gera gráficos
(PNG) comparando os datasets: acurácia, tempo de execução, número de
iterações e aproveitamento de rotulagem.

Como usar:
    python plot_results.py

    O script lista os CSVs disponíveis em results/ e pergunta qual você
    quer plotar (Enter usa o mais recente). Também dá pra pular a
    pergunta passando o arquivo direto:

        python plot_results.py --input results/experiment_results_20250816_143022_sem_comite.csv

Os gráficos de cada rodada são salvos em results/plots/<nome_do_csv>/,
assim rodar o plot pra arquivos diferentes nunca sobrescreve os gráficos
de uma rodada anterior.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")


def escolher_arquivo_csv(caminho_informado: str | None) -> Path:
    """Se um caminho foi passado por --input, usa ele. Senão, lista os
    CSVs disponíveis em results/ e pergunta interativamente."""
    if caminho_informado:
        caminho = Path(caminho_informado)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        return caminho

    candidatos = sorted(
        RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidatos:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em '{RESULTS_DIR}/'. Rode run_experiments.py primeiro."
        )

    print("Arquivos disponíveis em results/ (mais recente primeiro):")
    for i, c in enumerate(candidatos, start=1):
        print(f"  [{i}] {c.name}")

    try:
        escolha = input(
            f"\nDigite o número ou o nome do arquivo (Enter usa o mais recente: {candidatos[0].name}): "
        ).strip()
    except EOFError:
        escolha = ""

    if not escolha:
        return candidatos[0]

    if escolha.isdigit():
        idx = int(escolha) - 1
        if 0 <= idx < len(candidatos):
            return candidatos[idx]
        print("Número inválido, usando o mais recente.")
        return candidatos[0]

    caminho = Path(escolha)
    if not caminho.is_absolute() and not caminho.exists():
        caminho = RESULTS_DIR / caminho
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return caminho


def load_data(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Cada rodada agora gera seu próprio CSV (run_experiments.py não usa
    # mais append), mas mantém essa checagem por segurança, caso você
    # tenha juntado CSVs de rodadas diferentes manualmente.
    n_antes = len(df)
    df = df.drop_duplicates(subset="dataset", keep="last").copy()
    if len(df) < n_antes:
        print(f"Aviso: {n_antes - len(df)} linha(s) duplicada(s) por dataset foram descartadas (mantida a mais recente).")

    if "erro" in df.columns:
        n_erros = df["erro"].notna().sum()
        if n_erros > 0:
            print(f"Aviso: {n_erros} dataset(s) com erro foram ignorados nos gráficos.")
        df = df[df["erro"].isna()].copy()

    df["dataset_label"] = df["dataset"].str.replace(".csv", "", regex=False)
    return df.reset_index(drop=True)


def dados_para(df: pd.DataFrame, colunas: list[str], nome_grafico: str) -> pd.DataFrame:
    """Filtra linhas que têm todas as colunas necessárias pro gráfico,
    avisando quais datasets ficaram de fora sem derrubar os outros gráficos."""
    sub = df.dropna(subset=colunas).copy()
    faltando = set(df["dataset_label"]) - set(sub["dataset_label"])
    if faltando:
        print(f"Aviso: dataset(s) sem dado suficiente para '{nome_grafico}', pulando: {', '.join(sorted(faltando))}")
    return sub.reset_index(drop=True)


ESPECIALISTA_CORES = {
    "KNN": "#6366f1",
    "Naive Bayes": "#f59e0b",
    "Decision Tree": "#10b981",
    "Random Forest": "#2563eb",
    "XGBoost": "#ef4444",
    "Logistic Regression": "#8b5cf6",
    "Neural Network": "#ec4899",
}


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), ha="right")


def anotar_especialista(ax, df, x_positions=None):
    """Adiciona o nome do especialista como anotação abaixo de cada barra."""
    if "melhor_modelo" not in df.columns:
        return
    if x_positions is None:
        x_positions = range(len(df))
    for xi, modelo in zip(x_positions, df["melhor_modelo"]):
        if pd.notna(modelo):
            ax.annotate(
                modelo,
                xy=(xi, 0),
                xytext=(0, -32),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.5,
                color=ESPECIALISTA_CORES.get(modelo, "#475569"),
                rotation=45,
                clip_on=False,
            )


def plot_acuracia(df: pd.DataFrame, output_dir: Path):
    df = dados_para(df, ["acuracia_antes_self_training", "acuracia_apos_self_training"], "acurácia")
    if df.empty:
        print("Nenhum dataset com dados de acurácia, gráfico não gerado.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))

    x = range(len(df))
    largura = 0.35
    ax.bar(
        [i - largura / 2 for i in x],
        df["acuracia_antes_self_training"],
        width=largura,
        label="Acurácia inicial (treino c/ poucos dados)",
        color="#94a3b8",
    )
    ax.bar(
        [i + largura / 2 for i in x],
        df["acuracia_apos_self_training"],
        width=largura,
        label="Acurácia final (especialista após self-training)",
        color="#2563eb",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["dataset_label"])
    ax.set_ylim(0, 1)
    ax.legend()
    style_axis(ax, "Acurácia por dataset: inicial vs. após self-training", "Acurácia")
    anotar_especialista(ax, df, x)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "acuracia_por_dataset.png", dpi=150)
    plt.close(fig)


def plot_tempo(df: pd.DataFrame, output_dir: Path):
    df = dados_para(df, ["tempo_self_training_s", "tempo_total_s"], "tempo de execução")
    if df.empty:
        print("Nenhum dataset com dados de tempo, gráfico não gerado.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))

    x = range(len(df))
    ax.bar(x, df["tempo_self_training_s"], label="Self-training", color="#f59e0b")
    ax.bar(
        x,
        df["tempo_total_s"] - df["tempo_self_training_s"],
        bottom=df["tempo_self_training_s"],
        label="Resto do pipeline (seleção de modelo + comitê)",
        color="#94a3b8",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["dataset_label"])
    ax.legend()
    style_axis(ax, "Tempo de execução por dataset", "Tempo (segundos)")
    anotar_especialista(ax, df, x)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "tempo_execucao_por_dataset.png", dpi=150)
    plt.close(fig)


def plot_iteracoes(df: pd.DataFrame, output_dir: Path):
    df = dados_para(df, ["n_iteracoes", "criterio_parada"], "número de iterações")
    if df.empty:
        print("Nenhum dataset com dados de iterações, gráfico não gerado.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))

    cores = df["criterio_parada"].map(
        {"all_labeled": "#16a34a", "max_iter": "#dc2626", "no_change": "#f59e0b"}
    ).fillna("#94a3b8")

    ax.bar(range(len(df)), df["n_iteracoes"], color=cores)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["dataset_label"])
    style_axis(ax, "Número de iterações até parar (por dataset)", "Iterações")
    anotar_especialista(ax, df, range(len(df)))

    from matplotlib.patches import Patch
    legenda = [
        Patch(color="#16a34a", label="all_labeled (convergiu)"),
        Patch(color="#dc2626", label="max_iter (não convergiu)"),
        Patch(color="#f59e0b", label="no_change (travou sem mudança)"),
    ]
    ax.legend(handles=legenda)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "iteracoes_por_dataset.png", dpi=150)
    plt.close(fig)


def plot_rotulagem(df: pd.DataFrame, output_dir: Path):
    df = dados_para(df, ["instancias_nao_rotuladas_antes", "instancias_nao_rotuladas_depois"], "rotulagem")
    if df.empty:
        print("Nenhum dataset com dados de rotulagem, gráfico não gerado.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))

    x = range(len(df))
    largura = 0.35
    ax.bar(
        [i - largura / 2 for i in x],
        df["instancias_nao_rotuladas_antes"],
        width=largura,
        label="Não rotuladas antes",
        color="#94a3b8",
    )
    ax.bar(
        [i + largura / 2 for i in x],
        df["instancias_nao_rotuladas_depois"],
        width=largura,
        label="Não rotuladas depois (sobraram)",
        color="#dc2626",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["dataset_label"])
    ax.legend()
    style_axis(ax, "Instâncias não rotuladas: antes vs. depois do self-training", "Nº de instâncias")
    anotar_especialista(ax, df, x)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / "rotulagem_por_dataset.png", dpi=150)
    plt.close(fig)


def plot_especialista(df: pd.DataFrame, output_dir: Path):
    """Dois painéis:
    1. Frequência de cada especialista (barras horizontais)
    2. Acurácia do especialista por dataset, com cor por modelo
    """
    if "melhor_modelo" not in df.columns:
        print("Coluna 'melhor_modelo' ausente, gráfico de especialista não gerado.")
        return

    sub = df.dropna(subset=["melhor_modelo"]).copy()
    if sub.empty:
        print("Nenhum dado de especialista para plotar.")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(max(14, len(sub) * 0.7 + 6), 6)
    )

    # ── Painel 1: frequência de cada modelo ─────────────────────────────────
    contagem = sub["melhor_modelo"].value_counts()
    cores_freq = [ESPECIALISTA_CORES.get(m, "#94a3b8") for m in contagem.index]
    axes[0].barh(contagem.index, contagem.values, color=cores_freq)
    axes[0].set_title("Frequência do especialista escolhido", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Nº de datasets")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(contagem.values):
        axes[0].text(v + 0.05, i, str(v), va="center", fontsize=10)

    # ── Painel 2: acurácia por dataset, cor = especialista ──────────────────
    cores_ds = [ESPECIALISTA_CORES.get(m, "#94a3b8") for m in sub["melhor_modelo"]]
    x = range(len(sub))
    bars = axes[1].bar(x, sub["acuracia_apos_self_training"], color=cores_ds, edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(sub["dataset_label"], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Acurácia do especialista por dataset", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_xlabel("Dataset")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # Anotação do nome do modelo dentro/acima da barra
    for bar, modelo in zip(bars, sub["melhor_modelo"]):
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            modelo,
            ha="center", va="bottom",
            fontsize=6, rotation=90,
            color=ESPECIALISTA_CORES.get(modelo, "#334155"),
        )

    # Legenda de cores (modelos)
    from matplotlib.patches import Patch
    handles = [
        Patch(color=cor, label=modelo)
        for modelo, cor in ESPECIALISTA_CORES.items()
        if modelo in sub["melhor_modelo"].values
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")

    fig.suptitle("Análise do especialista por dataset", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "especialista_por_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Gera gráficos a partir de um CSV de resultados.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Caminho do CSV a plotar (ex: results/experiment_results_20250816_143022.csv). "
        "Se não passar, o script lista os arquivos de results/ e pergunta interativamente.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        input_csv = escolher_arquivo_csv(args.input)
    except FileNotFoundError as exc:
        print(exc)
        return

    # Cada CSV plotado ganha sua própria subpasta de gráficos, então plotar
    # arquivos diferentes nunca sobrescreve os gráficos de uma rodada anterior.
    output_dir = PLOTS_DIR / input_csv.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPlotando: {input_csv}")

    df = load_data(input_csv)
    if df.empty:
        print("Nenhum resultado válido para plotar (todos os datasets deram erro?).")
        return

    plot_acuracia(df, output_dir)
    plot_tempo(df, output_dir)
    plot_iteracoes(df, output_dir)
    plot_rotulagem(df, output_dir)
    plot_especialista(df, output_dir)

    print(f"\nGráficos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()