"""Passo 1 do Artigo 05: SHAP global na CNN-1D do SecFedNet (Artigo 01).

Explica o logit de DDoS do checkpoint oficial (secfednet-cnn1d-v1) sobre fluxos do treino final
reproduzido por preparar_dados.py. O teste cronológico congelado do Artigo 01 não é usado.

Saídas em <saida>/:
  importancia_features.csv   ranking das 81 features (DeepSHAP, por classe, GradientSHAP, permutação)
  importancia_familias.csv   importância agregada por família de features
  confianca.json             aditividade, estabilidade entre sementes e concordância entre métodos
  figuras/*.png              gráficos para o artigo

Uso:
    python shap_global.py <repo_secfednet> <dados_treino.npz> <checkpoint_cnn1d.pt> <saida> [--n 2000] [--fundo 500]
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr

CNN1D_SHA256 = "ed4f90225d3bbb765f9dc3bf461362ba7bdae6c47e70ac1c3333bbe8b0598861"
CHECKPOINT_SHA256 = "37db7d4f6d0123786cc4ed5227d670edb1344c07089e8ff6910c6446e57400f3"


def carregar_cnn1d(repo):
    """Importa a CNN1D do próprio Artigo 01, conferindo que o código é o do contrato."""
    caminho = repo / "secfednet-local" / "artigo-01" / "cnn-flower" / "cnn1d.py"
    assert hashlib.sha256(caminho.read_bytes()).hexdigest() == CNN1D_SHA256, "cnn1d.py diferente do contrato"
    spec = importlib.util.spec_from_file_location("cnn1d", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.CNN1D

FAMILIAS = {
    "Duração do fluxo": ["Flow Duration", "Total TCP Flow Time"],
    "Contagem de pacotes": ["Total Fwd Packet", "Total Bwd packets", "Subflow Fwd Packets",
                            "Subflow Bwd Packets", "Fwd Act Data Pkts"],
    "Volume (bytes)": ["Total Length of Fwd Packet", "Total Length of Bwd Packet",
                       "Subflow Fwd Bytes", "Subflow Bwd Bytes"],
    "Taxas (por segundo)": ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
                            "Down/Up Ratio"],
    "Flags TCP": ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd RST Flags",
                  "Bwd RST Flags", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
                  "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count"],
    "Cabeçalho e janela TCP": ["Fwd Header Length", "Bwd Header Length", "FWD Init Win Bytes",
                               "Bwd Init Win Bytes"],
    "Bulk": ["Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
             "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg"],
    "Atividade e ociosidade": ["Active Mean", "Active Std", "Active Max", "Active Min",
                               "Idle Mean", "Idle Std", "Idle Max", "Idle Min"],
    "ICMP": ["ICMP Code", "ICMP Type"],
}
PREFIXOS = {"Tempo entre pacotes (IAT)": ("Flow IAT", "Fwd IAT", "Bwd IAT"),
            "Tamanho de pacote": ("Fwd Packet Length", "Bwd Packet Length", "Packet Length",
                                  "Average Packet Size", "Fwd Segment Size", "Bwd Segment Size",
                                  "Fwd Seg Size")}


def familia(nome):
    for fam, nomes in FAMILIAS.items():
        if nome in nomes:
            return fam
    for fam, prefixos in PREFIXOS.items():
        if nome.startswith(prefixos):
            return fam
    raise KeyError(nome)


class Logit(torch.nn.Module):
    """A CNN devolve (logit, embedding); o SHAP precisa só do logit, com forma (n, 1)."""

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        return self.modelo(x)[0].unsqueeze(1)


def logits(f, x, lote=8192):
    with torch.no_grad():
        return np.concatenate([f(torch.from_numpy(x[i:i + lote])).squeeze(1).numpy()
                               for i in range(0, len(x), lote)])


def amostrar(y, n, fundo, seed):
    rng = np.random.default_rng(seed)
    ddos, normal = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    i_fundo = rng.choice(normal, fundo, replace=False)
    restantes = np.setdiff1d(normal, i_fundo)
    i_amostra = np.r_[rng.choice(ddos, n // 2, replace=False), rng.choice(restantes, n // 2, replace=False)]
    return i_fundo, i_amostra


def deep_shap(f, x, i_fundo, i_amostra):
    explicador = shap.DeepExplainer(f, torch.from_numpy(x[i_fundo]))
    valores = explicador.shap_values(torch.from_numpy(x[i_amostra]), check_additivity=True)
    base = float(np.asarray(explicador.expected_value).ravel()[0])
    return np.asarray(valores)[..., 0], base


def importancia_permutacao(f, x, y, seed, n=20000):
    """Aumento da log-loss ao embaralhar cada feature (média de 3 repetições)."""
    rng = np.random.default_rng(seed)
    idx = np.r_[rng.choice(np.flatnonzero(y == 1), n // 2, replace=False),
                rng.choice(np.flatnonzero(y == 0), n // 2, replace=False)]
    xs, ys = x[idx].copy(), y[idx]

    def logloss(xx):
        p = 1 / (1 + np.exp(-logits(f, xx).astype(np.float64)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p)))

    referencia = logloss(xs)
    aumento = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        original = xs[:, j].copy()
        for _ in range(3):
            xs[:, j] = rng.permutation(original)
            aumento[j] += (logloss(xs) - referencia) / 3
        xs[:, j] = original
    return aumento


def sobreposicao_top(a, b, k=10):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path, help="raiz do repositório secfednet")
    p.add_argument("dados", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("saida", type=Path)
    p.add_argument("--n", type=int, default=2000, help="fluxos explicados (metade DDoS, metade normais)")
    p.add_argument("--fundo", type=int, default=500, help="fluxos normais usados como linha de base")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = p.parse_args()
    (a.saida / "figuras").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    d = np.load(a.dados)
    x, y, colunas = d["x"], d["y"], [str(c) for c in d["colunas"]]
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256, "checkpoint diferente"
    CNN1D = carregar_cnn1d(a.repo)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    modelo = CNN1D()
    modelo.load_state_dict(ck["state_dict"])
    f = Logit(modelo).eval()

    # 1. DeepSHAP em várias sementes (fundo e amostra mudam a cada semente)
    rodadas = []
    for seed in a.seeds:
        i_fundo, i_amostra = amostrar(y, a.n, a.fundo, seed)
        sv, base = deep_shap(f, x, i_fundo, i_amostra)
        erro = np.abs(sv.sum(1) + base - logits(f, x[i_amostra])).max()
        rodadas.append({"seed": seed, "sv": sv, "base": base, "i": i_amostra, "erro_aditividade": float(erro)})
        print(f"seed {seed}: DeepSHAP ok, erro máximo de aditividade {erro:.2e}", flush=True)
    principal = rodadas[0]
    sv, ia = principal["sv"], principal["i"]
    imp = np.abs(sv).mean(0)
    imp_ddos = np.abs(sv[y[ia] == 1]).mean(0)
    imp_normal = np.abs(sv[y[ia] == 0]).mean(0)
    media_sinal_ddos = sv[y[ia] == 1].mean(0)

    # 2. Segunda opinião: GradientSHAP e permutação
    i_fundo, _ = amostrar(y, a.n, a.fundo, a.seeds[0])
    ge = shap.GradientExplainer(f, torch.from_numpy(x[i_fundo]))
    sg = np.asarray(ge.shap_values(torch.from_numpy(x[ia]), nsamples=200, rseed=0))[..., 0]
    imp_grad = np.abs(sg).mean(0)
    print("GradientSHAP ok", flush=True)
    imp_perm = importancia_permutacao(f, x, y, seed=a.seeds[0])
    print("Permutação ok", flush=True)

    # 3. Tabelas
    tab = pd.DataFrame({
        "feature": colunas, "familia": [familia(c) for c in colunas],
        "shap_medio_abs": imp, "shap_ddos": imp_ddos, "shap_normal": imp_normal,
        "shap_ddos_sinal_medio": media_sinal_ddos, "gradient_shap": imp_grad, "permutacao_logloss": imp_perm,
    }).sort_values("shap_medio_abs", ascending=False)
    tab.insert(0, "posicao", range(1, len(tab) + 1))
    tab["fracao_do_total"] = tab["shap_medio_abs"] / tab["shap_medio_abs"].sum()
    tab["fracao_acumulada"] = tab["fracao_do_total"].cumsum()
    tab.to_csv(a.saida / "importancia_features.csv", index=False, float_format="%.6g")

    fam = (tab.groupby("familia")
           .agg(shap_medio_abs=("shap_medio_abs", "sum"), n_features=("feature", "size"),
                feature_principal=("feature", "first"))
           .sort_values("shap_medio_abs", ascending=False))
    fam["fracao_do_total"] = fam["shap_medio_abs"] / fam["shap_medio_abs"].sum()
    fam.to_csv(a.saida / "importancia_familias.csv", float_format="%.6g")

    # 4. Confiança
    estab = []
    for r in rodadas[1:]:
        outra = np.abs(r["sv"]).mean(0)
        estab.append({"seeds": [principal["seed"], r["seed"]],
                      "spearman": float(spearmanr(imp, outra).statistic),
                      "top10_em_comum": sobreposicao_top(imp, outra)})
    confianca = {
        "explicador": "DeepExplainer (DeepLIFT/SHAP) sobre o logit de DDoS",
        "fluxos_explicados": int(a.n), "fundo_normais": int(a.fundo),
        "origem_dos_fluxos": "treino final reproduzido do Artigo 01 (teste congelado não usado)",
        "logit_base": principal["base"],
        "aditividade_erro_maximo": max(r["erro_aditividade"] for r in rodadas),
        "estabilidade_entre_sementes": estab,
        "deep_vs_gradient": {"spearman": float(spearmanr(imp, imp_grad).statistic),
                             "top10_em_comum": sobreposicao_top(imp, imp_grad)},
        "deep_vs_permutacao": {"spearman": float(spearmanr(imp, imp_perm).statistic),
                               "top10_em_comum": sobreposicao_top(imp, imp_perm)},
        "features_para_50pct": int((tab["fracao_acumulada"] < 0.5).sum() + 1),
        "features_para_80pct": int((tab["fracao_acumulada"] < 0.8).sum() + 1),
    }
    (a.saida / "confianca.json").write_text(json.dumps(confianca, ensure_ascii=False, indent=2) + "\n")

    # 5. Figuras
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    top = tab.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["shap_medio_abs"], color="#378ADD")
    ax.set_xlabel("|SHAP| médio (logit de DDoS)")
    ax.set_title("Top 20 features da CNN-1D do SecFedNet")
    fig.tight_layout(); fig.savefig(a.saida / "figuras" / "top20_features.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ff = fam.iloc[::-1]
    ax.barh(ff.index, ff["fracao_do_total"] * 100, color="#1D9E75")
    for yv, (v, n) in enumerate(zip(ff["fracao_do_total"] * 100, ff["n_features"])):
        ax.text(v + 0.5, yv, f"{v:.1f}%  ({n} features)", va="center", fontsize=8)
    ax.set_xlabel("% da importância total")
    ax.set_title("Importância por família de features")
    ax.set_xlim(0, ff["fracao_do_total"].max() * 100 * 1.35)
    fig.tight_layout(); fig.savefig(a.saida / "figuras" / "familias.png"); plt.close(fig)

    expl = shap.Explanation(values=sv[y[ia] == 1], base_values=np.full((y[ia] == 1).sum(), principal["base"]),
                            data=x[ia][y[ia] == 1], feature_names=colunas)
    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.title("Fluxos DDoS: como cada feature empurra o logit")
    plt.gcf().set_size_inches(8, 6); plt.tight_layout()
    plt.savefig(a.saida / "figuras" / "beeswarm_ddos.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(imp / imp.max(), imp_perm / imp_perm.max(), s=14, color="#7F77DD")
    for j in np.argsort(-imp)[:6]:
        ax.annotate(colunas[j], (imp[j] / imp.max(), imp_perm[j] / imp_perm.max()), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("DeepSHAP (normalizado)"); ax.set_ylabel("Permutação (normalizado)")
    ax.set_title(f"Concordância entre métodos (Spearman {confianca['deep_vs_permutacao']['spearman']:.2f})")
    fig.tight_layout(); fig.savefig(a.saida / "figuras" / "deep_vs_permutacao.png"); plt.close(fig)

    print(json.dumps(confianca, ensure_ascii=False, indent=2))
    print(tab.head(15)[["posicao", "feature", "familia", "shap_medio_abs", "fracao_acumulada",
                        "shap_ddos_sinal_medio"]].to_string(index=False))
    print(fam.to_string())


if __name__ == "__main__":
    main()
