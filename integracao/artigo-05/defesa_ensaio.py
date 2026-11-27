"""Passo 4 do Artigo 05 (ensaio): corrigir o ponto cego e medir se a evasão cai.

Parte do global do Artigo 01 e faz mais 3 rodadas de FedAvg nos mesmos 5 clientes (Adam lr=0,001
reiniciado a cada rodada, 1 época local, lote 512, BCE), em duas variantes:
  controle     treino extra sem nada novo (separa o efeito de "treinar mais")
  adversarial  cada cliente acrescenta cópias disfarçadas dos seus próprios DDoS: um grupo de features
               trocado, numa fração alfa ~ U(0,5; 1), por valores de BENIGN do próprio cliente.
               Duas versões: "top-8" (as 8 mais importantes do SHAP) e "família" (as 21 features de
               tamanho de pacote e volume)
O cliente 2 não tem DDoS e treina sem cópias. 10% do treino ficam reservados para avaliação e não
entram em nenhum treino extra (o global original já os tinha visto, por isso o desempenho limpo dele
é otimista). O teste de sexta continua reservado.

Avaliação: desempenho limpo, evasão nos cenários do Passo 3 e um ATAQUE ADAPTATIVO, em que o atacante
refaz o SHAP no modelo corrigido e troca as novas top-k features.

Uso:
    python defesa_ensaio.py <repo_secfednet> <dados_treino.npz> <pasta_pacote_shap> <resultados>
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
from torch import nn

from evasao_ensaio import transplantar
from shap_clientes import MODELOS, sha
from shap_global import Logit, amostrar, carregar_cnn1d, deep_shap, familia, logits

SEED = 11
RODADAS, LOTE, TAXA = 3, 512, 0.001
FRACAO_AVALIACAO = 0.10
N_ATAQUE = 2000
KS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]


def treinar_local(modelo, x, y, seed):
    torch.manual_seed(seed)
    otim = torch.optim.Adam(modelo.parameters(), lr=TAXA)
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    modelo.train()
    for lote in torch.randperm(len(xt), generator=g).split(LOTE):
        otim.zero_grad()
        perda = bce(modelo(xt[lote])[0], yt[lote])
        perda.backward()
        otim.step()
    modelo.eval()


def fedavg(CNN1D, estado_inicial, dados_clientes, cols, rng, seed_base):
    global_estado = {k: v.clone() for k, v in estado_inicial.items()}
    historico = []
    for rodada in range(1, RODADAS + 1):
        estados, pesos = [], []
        for i, (xc, yc) in enumerate(dados_clientes):
            if cols is not None and (yc == 1).any():
                ddos, normais = xc[yc == 1], xc[yc == 0]
                doadores = normais[rng.integers(0, len(normais), len(ddos))]
                alfa = rng.uniform(0.5, 1.0, (len(ddos), 1)).astype(np.float32)
                copia = ddos.copy()
                copia[:, cols] = (1 - alfa) * ddos[:, cols] + alfa * doadores[:, cols]
                xc, yc = np.concatenate([xc, copia]), np.concatenate([yc, np.ones(len(copia), dtype=yc.dtype)])
            m = CNN1D(); m.load_state_dict(global_estado)
            treinar_local(m, xc, yc, seed_base + 100 * rodada + i)
            estados.append(m.state_dict()); pesos.append(len(xc))
        total = sum(pesos)
        global_estado = {k: sum(e[k] * (w / total) for e, w in zip(estados, pesos)) for k in global_estado}
        historico.append({"rodada": rodada, "exemplos": total})
    m = CNN1D(); m.load_state_dict(global_estado)
    return m, historico


def desempenho(f, x, y):
    prev = logits(f, x) >= 0
    return {"recall_ddos": float(prev[y == 1].mean()), "taxa_falso_alarme": float(prev[y == 0].mean()),
            "falsos_alarmes": int(prev[y == 0].sum()), "ddos_perdidos": int((~prev[y == 1]).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path)
    p.add_argument("dados", type=Path)
    p.add_argument("pacote", type=Path)
    p.add_argument("resultados", type=Path)
    a = p.parse_args()
    saida = a.resultados / "defesa"
    (saida / "figuras").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    d = np.load(a.dados)
    x, y, cliente, colunas = d["x"], d["y"], d["cliente"], [str(c) for c in d["colunas"]]
    ranking = pd.read_csv(a.resultados / "importancia_features.csv")
    pos = {c: i for i, c in enumerate(colunas)}
    ordem_original = [pos[c] for c in ranking["feature"]]
    top8 = ordem_original[:8]

    CNN1D = carregar_cnn1d(a.repo)
    arquivo, esperado = MODELOS["Global"]
    assert sha(a.pacote / arquivo) == esperado
    estado = torch.load(a.pacote / arquivo, map_location="cpu", weights_only=True)["state_dict"]
    original = CNN1D(); original.load_state_dict(estado)

    # Avaliação: 10% estratificado, fora de qualquer treino extra
    rng = np.random.default_rng(SEED)
    aval = np.concatenate([rng.choice(np.flatnonzero(y == c), int((y == c).sum() * FRACAO_AVALIACAO), replace=False)
                           for c in (0, 1)])
    em_aval = np.zeros(len(y), bool); em_aval[aval] = True
    dados_clientes = [(x[(cliente == i) & ~em_aval], y[(cliente == i) & ~em_aval]) for i in range(5)]
    print("Treino extra por cliente:", [len(c[1]) for c in dados_clientes], "| avaliação:", len(aval), flush=True)

    familia_21 = [pos[c] for c in colunas if familia(c) in ("Tamanho de pacote", "Volume (bytes)")]
    variantes = {"Controle (treino extra)": None, "Adversarial top-8": top8, "Adversarial família": familia_21}
    modelos = {"Original": Logit(original).eval()}
    historicos = {}
    for k, (nome, cols) in enumerate(variantes.items()):
        m, h = fedavg(CNN1D, estado, dados_clientes, cols, np.random.default_rng(SEED + k), 2300 + 50 * k)
        modelos[nome] = Logit(m).eval(); historicos[nome] = h
        print(f"{nome}: treinado", flush=True)

    xa, ya = x[aval], y[aval]
    limpo = {n: desempenho(f, xa, ya) for n, f in modelos.items()}

    # Fluxos de ataque: DDoS da avaliação que o original detecta; doadores BENIGN da avaliação
    ddos = np.flatnonzero(ya == 1)
    ddos = ddos[logits(modelos["Original"], xa[ddos]) >= 0]
    i_ddos = rng.choice(ddos, min(N_ATAQUE, len(ddos)), replace=False)
    i_doador = rng.choice(np.flatnonzero(ya == 0), len(i_ddos), replace=False)
    xd, xb = xa[i_ddos], xa[i_doador]

    def idx_familia(*fams):
        return [pos[c] for c in colunas if familia(c) in fams]

    cenarios = {"Top-4 SHAP (original)": ordem_original[:4], "Top-8 SHAP (original)": top8,
                "Tamanho de pacote (17)": idx_familia("Tamanho de pacote"),
                "Tamanho de pacote + Volume (21)": idx_familia("Tamanho de pacote", "Volume (bytes)")}
    evasao = {c: {n: float((logits(f, transplantar(xd, xb, cols)) < 0).mean()) for n, f in modelos.items()}
              for c, cols in cenarios.items()}

    # Ataque adaptativo: SHAP refeito em cada modelo, troca das top-k dele
    i_fundo, i_amostra = amostrar(y, 2000, 500, 1)
    rank_modelo, familias_modelo, curva = {}, {}, []
    for n, f in modelos.items():
        sv, _ = deep_shap(f, x, i_fundo, i_amostra)
        imp = np.abs(sv).mean(0)
        rank_modelo[n] = list(np.argsort(-imp))
        fam = pd.Series(imp, index=colunas).groupby([familia(c) for c in colunas]).sum()
        familias_modelo[n] = (fam / fam.sum()).sort_values(ascending=False)
        for k in KS:
            ev = float((logits(f, transplantar(xd, xb, rank_modelo[n][:k])) < 0).mean())
            curva.append({"modelo": n, "k": k, "evasao": ev})
        print(f"{n}: ataque adaptativo ok", flush=True)
    curva = pd.DataFrame(curva)
    k_para = {n: next((int(r.k) for r in curva[curva.modelo == n].itertuples() if r.evasao >= 0.9), None)
              for n in modelos}

    resultado = {
        "natureza": "ensaio no espaço de features; treino FedAvg simulado em processo, sem Flower",
        "configuracao": {"rodadas_extras": RODADAS, "lote": LOTE, "taxa": TAXA, "epocas_locais": 1,
                         "fracao_avaliacao": FRACAO_AVALIACAO, "seed": SEED, "features_aumentadas": {n: (None if c is None else [colunas[i] for i in c]) for n, c in variantes.items()},
                         "alfa_aumento": "U(0,5; 1)", "fluxos_de_ataque": int(len(i_ddos))},
        "desempenho_limpo_na_avaliacao": limpo,
        "evasao_cenarios_fixos": evasao,
        "ataque_adaptativo": {"k_para_90pct_de_evasao": k_para,
                              "top8_de_cada_modelo": {n: [colunas[i] for i in r[:8]] for n, r in rank_modelo.items()}},
        "familias_por_modelo": {n: s.round(4).to_dict() for n, s in familias_modelo.items()},
        "historico": historicos,
    }
    (saida / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    curva.to_csv(saida / "curva_adaptativa.csv", index=False, float_format="%.4f")
    pd.DataFrame(evasao).T.to_csv(saida / "evasao_cenarios.csv", float_format="%.4f", index_label="cenario")
    pd.DataFrame(familias_modelo).to_csv(saida / "familias_por_modelo.csv", float_format="%.4f", index_label="familia")

    # Figuras
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    cores = {"Original": "#444441", "Controle (treino extra)": "#B4B2A9", "Adversarial top-8": "#5DCAA5",
             "Adversarial família": "#0F6E56"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    nomes_c = list(cenarios)
    larg = 0.2
    for j, (n, cor) in enumerate(cores.items()):
        vals = [evasao[c][n] * 100 for c in nomes_c]
        xs = np.arange(len(nomes_c)) + (j - 1.5) * larg
        ax.bar(xs, vals, larg, color=cor, label=n)
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(nomes_c)), [c.replace(" (", "\n(") for c in nomes_c], fontsize=8)
    ax.set_ylabel("% dos DDoS que passam como BENIGN"); ax.set_ylim(0, 108)
    ax.set_title("Evasão antes e depois da correção (ataques do Passo 3)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(); fig.savefig(saida / "figuras" / "evasao_antes_depois.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for n, cor in cores.items():
        c = curva[curva.modelo == n]
        ax.plot(c["k"], c["evasao"] * 100, marker="o", ms=3.5, color=cor, label=n)
    ax.set_xscale("log", base=2); ax.set_xticks(KS, KS)
    ax.axhline(90, color="#B4B2A9", lw=0.6, ls="--")
    ax.set_xlabel("Features trocadas pelo atacante (top-k do SHAP de cada modelo)")
    ax.set_ylabel("Evasão (%)")
    ax.set_title("Ataque adaptativo: o atacante refaz o SHAP no modelo corrigido")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(saida / "figuras" / "ataque_adaptativo.png"); plt.close(fig)

    print(json.dumps({k: resultado[k] for k in ("desempenho_limpo_na_avaliacao", "evasao_cenarios_fixos",
                                                "ataque_adaptativo")}, ensure_ascii=False, indent=1))
    print(curva.pivot(index="k", columns="modelo", values="evasao").round(3).to_string())
    print(pd.DataFrame(familias_modelo).round(3).to_string())


if __name__ == "__main__":
    main()
