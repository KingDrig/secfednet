"""Passo 2 do Artigo 05: SHAP por cliente federado.

Duas perguntas:
  A. O que cada modelo aprendeu? O global e os 5 locais (rodada 3, pós-fit, pré-agregação) são
     explicados sobre os MESMOS fluxos e o MESMO fundo do Passo 1. As diferenças vêm só dos modelos.
  B. O global olha para os dados de cada cliente do mesmo jeito? O modelo global é explicado sobre
     uma amostra de cada cliente.

Saídas em <saida>/clientes/:
  importancia_por_modelo.csv     |SHAP| médio por feature e modelo (A)
  familias_por_modelo.csv        % da importância por família e modelo (A)
  divergencia.json               Spearman e top-10 em comum entre modelos, agregação, desempenho
  global_por_cliente.csv         |SHAP| médio do global sobre os dados de cada cliente (B)
  figuras/*.png

Uso:
    python shap_clientes.py <repo_secfednet> <dados_treino.npz> <pasta_pacote_shap> <saida>
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from shap_global import Logit, amostrar, carregar_cnn1d, deep_shap, familia, logits, sobreposicao_top

PACOTE_SHA256 = "9eac2d389dbbc6e7e4eec030507b896fe51be4f634d0f6c0b8903a079b5bab92"
MODELOS = {
    "Global": ("modelos/global_zip.pt", "37db7d4f6d0123786cc4ed5227d670edb1344c07089e8ff6910c6446e57400f3"),
    "Cliente 0": ("modelos/cliente_0_rodada_03.pt", "ba3b6847584e9270d83c3fafd8ee3247c85961998b7aeec45ca7920b07bc92da"),
    "Cliente 1": ("modelos/cliente_1_rodada_03.pt", "08c9480e2867691e4c900308022638d2f73d99949bd86dfd9fbc40968642150e"),
    "Cliente 2": ("modelos/cliente_2_rodada_03.pt", "0596f1eff0fdef040d3e8a1f5b32da1909dba0317986d60f56f7f6e3c7e482fe"),
    "Cliente 3": ("modelos/cliente_3_rodada_03.pt", "f9c4a629eca53bb986ec32e16816862143cb854ca0e4c5222cc957a534f11338"),
    "Cliente 4": ("modelos/cliente_4_rodada_03.pt", "722f02b9e1838e0ca147f346a2a2d1472d5e7e9f05928d24584530cf9ef81d82"),
}
SEED = 1          # mesma amostra e mesmo fundo do Passo 1
N, FUNDO = 2000, 500
N_POR_CLIENTE = 500  # por classe, quando o cliente tem


def sha(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def desempenho(f, x, y):
    previsto = logits(f, x) >= 0
    vp = int((previsto & (y == 1)).sum())
    fp = int((previsto & (y == 0)).sum())
    return {"acerto": float((previsto == y).mean()), "recall_ddos": vp / int((y == 1).sum()),
            "taxa_falso_alarme": fp / int((y == 0).sum()), "falsos_alarmes": fp,
            "ddos_perdidos": int((y == 1).sum()) - vp}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path)
    p.add_argument("dados", type=Path)
    p.add_argument("pacote", type=Path, help="pasta extraída de secfednet-shap-fedavg3.zip")
    p.add_argument("saida", type=Path)
    a = p.parse_args()
    saida = a.saida / "clientes"
    (saida / "figuras").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    d = np.load(a.dados)
    x, y, cliente, colunas = d["x"], d["y"], d["cliente"], [str(c) for c in d["colunas"]]
    with np.load(a.pacote / "indices_particao.npz", allow_pickle=False) as idx:
        assert np.array_equal(idx["treino_csv_registro1"], d["linhas"])
        for i in range(5):
            assert np.array_equal(idx[f"cliente_{i}_treino_idx0"], np.flatnonzero(cliente == i))
    print("Partição do pacote = partição reconstruída.", flush=True)

    CNN1D = carregar_cnn1d(a.repo)
    modelos = {}
    for nome, (arquivo, esperado) in MODELOS.items():
        assert sha(a.pacote / arquivo) == esperado, f"hash diferente: {arquivo}"
        ck = torch.load(a.pacote / arquivo, map_location="cpu", weights_only=True)
        m = CNN1D()
        m.load_state_dict(ck["state_dict"], strict=True)
        modelos[nome] = Logit(m).eval()

    # A. Cada modelo sobre os mesmos fluxos
    i_fundo, i_amostra = amostrar(y, N, FUNDO, SEED)
    ya = y[i_amostra]
    imp, imp_ddos, aditividade, perf, intensidade, contrib_ddos = {}, {}, {}, {}, {}, {}
    for nome, f in modelos.items():
        sv, base = deep_shap(f, x, i_fundo, i_amostra)
        lg = logits(f, x[i_amostra])
        aditividade[nome] = float(np.abs(sv.sum(1) + base - lg).max())
        intensidade[nome] = {"logit_base": base, "logit_medio_ddos": float(lg[ya == 1].mean()),
                             "logit_medio_benign": float(lg[ya == 0].mean()),
                             "soma_shap_media_ddos": float(sv[ya == 1].sum(1).mean())}
        contrib_ddos[nome] = sv[ya == 1].mean(0)
        imp[nome] = np.abs(sv).mean(0)
        imp_ddos[nome] = np.abs(sv[ya == 1]).mean(0)
        perf[nome] = desempenho(f, x, y)
        print(f"{nome}: DeepSHAP ok (aditividade {aditividade[nome]:.1e}), "
              f"recall DDoS {perf[nome]['recall_ddos']:.4f}, falso alarme {perf[nome]['taxa_falso_alarme']:.4f}",
              flush=True)

    nomes = list(modelos)
    tab = pd.DataFrame(imp, index=colunas)
    tab.insert(0, "familia", [familia(c) for c in colunas])
    tab.sort_values("Global", ascending=False).to_csv(saida / "importancia_por_modelo.csv",
                                                      index_label="feature", float_format="%.6g")
    fam = tab.groupby("familia")[nomes].sum()
    fam = (fam / fam.sum()).sort_values("Global", ascending=False)
    fam.to_csv(saida / "familias_por_modelo.csv", float_format="%.4f")

    sp = pd.DataFrame([[spearmanr(imp[r], imp[c]).statistic for c in nomes] for r in nomes], index=nomes, columns=nomes)
    top = pd.DataFrame([[sobreposicao_top(imp[r], imp[c]) for c in nomes] for r in nomes], index=nomes, columns=nomes)
    sp_ddos = pd.DataFrame([[spearmanr(imp_ddos[r], imp_ddos[c]).statistic for c in nomes] for r in nomes],
                           index=nomes, columns=nomes)

    # Explicação do global x média ponderada das explicações locais (FedAvg média pesos, não explicações)
    pesos = np.array([(cliente == i).sum() for i in range(5)], dtype=float)
    media_local = sum(w * imp[f"Cliente {i}"] for i, w in enumerate(pesos / pesos.sum()))
    top10 = {n: [colunas[j] for j in np.argsort(-imp[n])[:10]] for n in nomes}

    # Features que algum local valoriza muito e o global quase ignora (conhecimento diluído na agregação)
    rank = {n: pd.Series(imp[n], index=colunas).rank(ascending=False) for n in nomes}
    diluidas = []
    for c in colunas:
        melhor = min(nomes[1:], key=lambda n: rank[n][c])
        if rank[melhor][c] <= 10 and rank["Global"][c] > 25:
            diluidas.append({"feature": c, "familia": familia(c), "posicao_global": int(rank["Global"][c]),
                             "melhor_local": melhor, "posicao_no_local": int(rank[melhor][c])})

    # B. Global sobre os dados de cada cliente (fundo fixo do Passo 1)
    rng = np.random.default_rng(SEED)
    linhas_b, contagem_b = {}, {}
    for i in range(5):
        partes = []
        for classe in (0, 1):
            disponiveis = np.setdiff1d(np.flatnonzero((cliente == i) & (y == classe)), i_fundo)
            if len(disponiveis):
                partes.append(rng.choice(disponiveis, min(N_POR_CLIENTE, len(disponiveis)), replace=False))
        ib = np.concatenate(partes)
        sv, _ = deep_shap(modelos["Global"], x, i_fundo, ib)
        linhas_b[f"Cliente {i}"] = np.abs(sv).mean(0)
        contagem_b[f"Cliente {i}"] = {"BENIGN": int((y[ib] == 0).sum()), "DDoS": int((y[ib] == 1).sum())}
    tab_b = pd.DataFrame(linhas_b, index=colunas)
    tab_b.insert(0, "Global (amostra comum)", imp["Global"])
    tab_b.sort_values("Global (amostra comum)", ascending=False).to_csv(
        saida / "global_por_cliente.csv", index_label="feature", float_format="%.6g")
    sp_b = {c: float(spearmanr(imp["Global"], tab_b[c]).statistic) for c in linhas_b}
    print("Parte B ok", flush=True)

    dif_c2 = pd.Series(contrib_ddos["Cliente 2"] - contrib_ddos["Global"], index=colunas).sort_values()
    pd.DataFrame({n: contrib_ddos[n] for n in nomes}, index=colunas).sort_values("Global", ascending=False).to_csv(
        saida / "contribuicao_media_ddos_por_modelo.csv", index_label="feature", float_format="%.6g")

    resultado = {
        "amostra_comum": {"fluxos": N, "fundo_benign": FUNDO, "seed": SEED, "origem": "treino final (teste reservado não usado)"},
        "aditividade_erro_maximo": aditividade,
        "desempenho_no_treino_completo": perf,
        "intensidade": intensidade,
        "cliente2_menos_global_contribuicao_ddos": {"mais_fracas": dif_c2.head(6).round(4).to_dict(),
                                                    "mais_fortes": dif_c2.tail(3).round(4).to_dict()},
        "spearman_entre_modelos": sp.round(4).to_dict(),
        "spearman_entre_modelos_so_ddos": sp_ddos.round(4).to_dict(),
        "top10_em_comum": top.to_dict(),
        "top10_por_modelo": top10,
        "global_vs_media_ponderada_dos_locais": {
            "spearman": float(spearmanr(imp["Global"], media_local).statistic),
            "top10_em_comum": sobreposicao_top(imp["Global"], media_local)},
        "features_diluidas_na_agregacao": diluidas,
        "parte_b_global_sobre_cada_cliente": {"amostras": contagem_b, "spearman_vs_amostra_comum": sp_b},
    }
    (saida / "divergencia.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")

    # Figuras
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(sp.values, cmap="Blues", vmin=0.85, vmax=1)
    ax.set_xticks(range(6), nomes, rotation=30, ha="right"); ax.set_yticks(range(6), nomes)
    for r in range(6):
        for c in range(6):
            v = sp.values[r, c]
            ax.text(c, r, f"{v:.2f}\n{top.values[r, c]}/10", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.95 else "black")
    ax.set_title("Quanto os modelos concordam no que é importante\n(Spearman e features do top-10 em comum; escala de 0,85 a 1)")
    fig.tight_layout(); fig.savefig(saida / "figuras" / "concordancia_modelos.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    mostra = fam.loc[fam.max(axis=1) >= 0.03]
    ax.imshow(mostra.values * 100, cmap="YlOrBr", aspect="auto", vmin=0)
    ax.set_xticks(range(6), nomes, rotation=30, ha="right"); ax.set_yticks(range(len(mostra)), mostra.index)
    for r in range(len(mostra)):
        for c in range(6):
            v = mostra.values[r, c] * 100
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white" if v > 35 else "black")
    ax.set_title("% da importância por família de features, em cada modelo")
    fig.tight_layout(); fig.savefig(saida / "figuras" / "familias_por_modelo.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cores = ["#444441"] + ["#378ADD"] * 5
    cores[3] = "#D85A30"
    vals = [intensidade[n]["logit_medio_ddos"] for n in nomes]
    ax.bar(nomes, vals, color=cores)
    for k, (v, n) in enumerate(zip(vals, nomes)):
        ax.text(k, v + 0.15, f"logit {v:.2f}\nrecall {perf[n]['recall_ddos'] * 100:.2f}%", ha="center", fontsize=7.5)
    ax.axhline(0, color="#888780", lw=0.8)
    ax.set_ylabel("Logit médio nos fluxos DDoS (limiar = 0)")
    ax.set_ylim(min(0, min(vals)) - 0.5, max(vals) * 1.3)
    ax.set_title("Mesmas features, intensidades diferentes: quão forte cada modelo acusa DDoS")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(saida / "figuras" / "intensidade_ddos.png"); plt.close(fig)

    print(json.dumps({k: resultado[k] for k in ("intensidade", "cliente2_menos_global_contribuicao_ddos",
                                                "global_vs_media_ponderada_dos_locais", "parte_b_global_sobre_cada_cliente",
                                                "features_diluidas_na_agregacao")}, ensure_ascii=False, indent=1))
    print(sp.round(2).to_string()); print(top.to_string()); print((fam * 100).round(1).to_string())
    for n in nomes:
        print(n, top10[n][:5])


if __name__ == "__main__":
    main()
