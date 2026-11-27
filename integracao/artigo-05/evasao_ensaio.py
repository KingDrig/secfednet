"""Passo 3 do Artigo 05: ensaio de evasão guiado pelo SHAP e mapa de blind spots.

Ensaio no espaço de features (não é tráfego real): para cada fluxo DDoS, os valores de um grupo de
features são trocados pelos de um fluxo BENIGN sorteado ("transplante"). Se o SHAP do Passo 1 está
certo, transplantar as features mais importantes deve enganar o detector, e transplantar as
irrelevantes não. Medido no global e nos 5 modelos locais.

Depois da evasão, o mapa de blind spots cruza, para cada feature, (a) o quanto ela ainda separa os
ataques disfarçados do tráfego normal (AUC) com (b) a importância SHAP. Feature que separa bem e
pesa pouco = sinal que denunciaria o ataque, mas que o modelo não usa.

Fluxos: do treino final (ensaio). O teste de sexta continua reservado; a versão final aguarda a
definição dos dados de evasão e os exemplos da GAN do Artigo 04.

Uso:
    python evasao_ensaio.py <repo_secfednet> <dados_treino.npz> <pasta_pacote_shap> <resultados>
(<resultados> deve conter importancia_features.csv do Passo 1)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from shap_clientes import MODELOS, sha
from shap_global import FAMILIAS, Logit, amostrar, carregar_cnn1d, familia, logits

SEED = 7
N_DDOS = 2000
N_SORTEIOS = 20
ALFAS = np.round(np.linspace(0, 1, 11), 2)


def evasao(modelos, x_adv):
    """Fração dos fluxos DDoS que passam a ser classificados como BENIGN (logit < 0)."""
    return {n: float((logits(f, x_adv) < 0).mean()) for n, f in modelos.items()}


def transplantar(x_ddos, x_doador, cols, alfa=1.0):
    adv = x_ddos.copy()
    adv[:, cols] = (1 - alfa) * x_ddos[:, cols] + alfa * x_doador[:, cols]
    return adv


def separacao(a, b):
    """|AUC - 0,5| x 2 por feature: 0 = indistinguível, 1 = separa perfeitamente."""
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    out = []
    for j in range(a.shape[1]):
        v = np.r_[a[:, j], b[:, j]]
        out.append(0.0 if np.ptp(v) == 0 else abs(roc_auc_score(y, v) - 0.5) * 2)
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path)
    p.add_argument("dados", type=Path)
    p.add_argument("pacote", type=Path)
    p.add_argument("resultados", type=Path)
    a = p.parse_args()
    saida = a.resultados / "evasao"
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    d = np.load(a.dados)
    x, y, colunas = d["x"], d["y"], [str(c) for c in d["colunas"]]
    ranking = pd.read_csv(a.resultados / "importancia_features.csv")
    assert set(ranking["feature"]) == set(colunas)
    pos = {c: i for i, c in enumerate(colunas)}
    ordem = [pos[c] for c in ranking["feature"]]           # do mais ao menos importante (Passo 1)
    imp = ranking.set_index("feature").loc[colunas, "shap_medio_abs"].to_numpy()

    CNN1D = carregar_cnn1d(a.repo)
    modelos = {}
    for nome, (arquivo, esperado) in MODELOS.items():
        assert sha(a.pacote / arquivo) == esperado
        ck = torch.load(a.pacote / arquivo, map_location="cpu", weights_only=True)
        m = CNN1D(); m.load_state_dict(ck["state_dict"]); modelos[nome] = Logit(m).eval()

    # Fluxos: DDoS que o global detecta; doadores BENIGN; separados da amostra/fundo do Passo 1
    i_fundo, i_amostra = amostrar(y, 2000, 500, 1)
    usados = np.union1d(i_fundo, i_amostra)
    rng = np.random.default_rng(SEED)
    ddos = np.setdiff1d(np.flatnonzero(y == 1), usados)
    ddos = ddos[logits(modelos["Global"], x[ddos]) >= 0]
    i_ddos = rng.choice(ddos, N_DDOS, replace=False)
    i_doador = rng.choice(np.setdiff1d(np.flatnonzero(y == 0), usados), N_DDOS, replace=False)
    xd, xb = x[i_ddos], x[i_doador]
    base = evasao(modelos, xd)
    print("Evasão sem ataque:", {k: round(v, 4) for k, v in base.items()}, flush=True)

    def idx_familia(*fams):
        return [pos[c] for c in colunas if familia(c) in fams]

    cenarios = {
        "Top-4 SHAP": ordem[:4],
        "Top-8 SHAP": ordem[:8],
        "Tamanho de pacote (17)": idx_familia("Tamanho de pacote"),
        "Tamanho de pacote + Volume (21)": idx_familia("Tamanho de pacote", "Volume (bytes)"),
        "Duração + Contagem (7)": idx_familia("Duração do fluxo", "Contagem de pacotes"),
        "Flags TCP (14)": idx_familia("Flags TCP"),
        "8 menos importantes": ordem[-8:],
        "Todas as 81": list(range(len(colunas))),
    }
    linhas = []
    for nome, cols in cenarios.items():
        ev = evasao(modelos, transplantar(xd, xb, cols))
        linhas.append({"cenario": nome, "n_features": len(cols),
                       "fracao_importancia_shap": float(imp[cols].sum() / imp.sum()), **ev})
    # Controle: 8 features sorteadas, média de N_SORTEIOS
    sorteios = [evasao(modelos, transplantar(xd, xb, list(rng.choice(len(colunas), 8, replace=False))))
                for _ in range(N_SORTEIOS)]
    linhas.append({"cenario": f"8 aleatórias (média de {N_SORTEIOS})", "n_features": 8,
                   "fracao_importancia_shap": float("nan"),
                   **{n: float(np.mean([s[n] for s in sorteios])) for n in modelos}})
    tab = pd.DataFrame(linhas)
    tab["media_locais"] = tab[[f"Cliente {i}" for i in range(5)]].mean(axis=1)
    tab.to_csv(saida / "taxas_evasao.csv", index=False, float_format="%.4f")
    print(tab[["cenario", "n_features", "fracao_importancia_shap", "Global", "media_locais",
               "Cliente 2"]].round(3).to_string(index=False), flush=True)

    # Curva de orçamento: quanto mover (alfa) as features em direção ao doador BENIGN
    curva = []
    for nome in ("Top-4 SHAP", "Top-8 SHAP", "Tamanho de pacote (17)", "8 menos importantes"):
        for alfa in ALFAS:
            ev = evasao({"Global": modelos["Global"]}, transplantar(xd, xb, cenarios[nome], alfa))
            curva.append({"cenario": nome, "alfa": float(alfa), "evasao_global": ev["Global"]})
    curva = pd.DataFrame(curva)
    curva.to_csv(saida / "curva_orcamento.csv", index=False, float_format="%.4f")

    # Mapa de blind spots, após o ataque mínimo que engana o global (Top-8 SHAP)
    adv = transplantar(xd, xb, cenarios["Top-8 SHAP"])
    passou = logits(modelos["Global"], adv) < 0
    sep_antes = separacao(xd, xb)
    sep_depois = separacao(adv[passou], xb) if passou.sum() >= 50 else np.full(len(colunas), np.nan)
    mapa = pd.DataFrame({"feature": colunas, "familia": [familia(c) for c in colunas],
                         "shap_medio_abs": imp, "posicao_shap": pd.Series(imp).rank(ascending=False).astype(int),
                         "separacao_antes": sep_antes, "separacao_depois_do_ataque": sep_depois,
                         "alterada_no_ataque": [i in set(cenarios["Top-8 SHAP"]) for i in range(len(colunas))]})
    limiar_imp = np.quantile(imp, 0.5)
    mapa["blind_spot"] = (~mapa["alterada_no_ataque"]) & (mapa["separacao_depois_do_ataque"] >= 0.8) & \
                         (mapa["shap_medio_abs"] <= limiar_imp)
    mapa.sort_values(["blind_spot", "separacao_depois_do_ataque"], ascending=False).to_csv(
        saida / "mapa_blind_spots.csv", index=False, float_format="%.4f")

    resumo = {
        "natureza": "ensaio no espaço de features (transplante de valores BENIGN); não é tráfego realizável",
        "fluxos_ddos": N_DDOS, "seed": SEED, "origem": "treino final, disjunto da amostra e do fundo do Passo 1",
        "evasao_sem_ataque": base,
        "ataque_do_mapa": "Top-8 SHAP", "fluxos_que_enganaram_o_global": int(passou.sum()),
        "limiar_importancia_baixa": float(limiar_imp), "criterio_blind_spot":
            "não alterada, separação depois do ataque >= 0,8 e |SHAP| abaixo da mediana",
        "blind_spots": mapa.loc[mapa["blind_spot"]].sort_values("separacao_depois_do_ataque", ascending=False)[
            ["feature", "familia", "posicao_shap", "separacao_depois_do_ataque"]].to_dict("records"),
    }
    (saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n")

    # Figuras
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    t = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    yy = np.arange(len(t))
    ax.barh(yy + 0.2, t["Global"] * 100, 0.4, color="#444441", label="Global")
    ax.barh(yy - 0.2, t["media_locais"] * 100, 0.4, color="#378ADD", label="Média dos 5 locais")
    ax.set_yticks(yy, [f"{c}" for c in t["cenario"]])
    for k, (g, l) in enumerate(zip(t["Global"], t["media_locais"])):
        ax.text(g * 100 + 1, k + 0.2, f"{g * 100:.0f}%", va="center", fontsize=7.5)
        ax.text(l * 100 + 1, k - 0.2, f"{l * 100:.0f}%", va="center", fontsize=7.5)
    ax.set_xlim(0, 112); ax.set_xlabel("% dos fluxos DDoS que passam como BENIGN")
    ax.set_title("Evasão ao trocar grupos de features por valores de tráfego normal")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(saida / "figuras" / "evasao_por_cenario.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    estilos = {"Top-4 SHAP": "#D85A30", "Top-8 SHAP": "#993C1D", "Tamanho de pacote (17)": "#378ADD",
               "8 menos importantes": "#888780"}
    for nome, cor in estilos.items():
        c = curva[curva["cenario"] == nome]
        ax.plot(c["alfa"] * 100, c["evasao_global"] * 100, marker="o", ms=3, color=cor, label=nome)
    ax.set_xlabel("Quanto as features são movidas em direção ao normal (%)")
    ax.set_ylabel("Evasão no modelo global (%)")
    ax.set_title("Quanto o atacante precisa mexer")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(saida / "figuras" / "curva_orcamento.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    m = mapa
    cor = np.where(m["alterada_no_ataque"], "#D85A30", np.where(m["blind_spot"], "#1D9E75", "#B4B2A9"))
    ax.scatter(m["shap_medio_abs"], m["separacao_depois_do_ataque"], c=cor, s=22)
    desloc = [(-6, 6, "right"), (8, 6, "left"), (8, -10, "left"), (8, 4, "left")]
    for k, (_, r) in enumerate(m[m["blind_spot"]].sort_values("separacao_depois_do_ataque", ascending=False)
                               .head(8).iterrows()):
        dx, dy, ha = desloc[k % len(desloc)]
        ax.annotate(r["feature"], (r["shap_medio_abs"], r["separacao_depois_do_ataque"]), fontsize=7,
                    xytext=(dx, dy), textcoords="offset points", ha=ha, color="#085041")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.axvline(limiar_imp, color="#888780", lw=0.6, ls="--"); ax.axhline(0.8, color="#888780", lw=0.6, ls="--")
    ax.set_xlabel("Importância para o modelo (|SHAP| médio, escala log)")
    ax.set_ylabel("Ainda separa ataque disfarçado de normal (0 a 1)")
    ax.set_title("Mapa de blind spots após o ataque Top-8 SHAP")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
                       (("#D85A30", "Alterada pelo atacante"), ("#1D9E75", "Blind spot: denuncia, mas pesa pouco"),
                        ("#B4B2A9", "Demais"))], frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.5, loc="center right")
    fig.tight_layout(); fig.savefig(saida / "figuras" / "mapa_blind_spots.png"); plt.close(fig)

    print(curva.pivot(index="alfa", columns="cenario", values="evasao_global").round(3).to_string())
    print(json.dumps({k: resumo[k] for k in ("fluxos_que_enganaram_o_global", "blind_spots")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
