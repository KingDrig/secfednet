"""Reproduz o treino final do Aluno 1 (Artigo 01) e a partição dos 5 clientes.

Reutiliza as funções do próprio Artigo 01 (ler_sexta, padronizar, distribuir_dirichlet) e confere
o resultado contra o hash do treino final e contra a média/escala gravadas no checkpoint.
O teste cronológico congelado só é lido porque o expurgo do Artigo 01 depende dele; nenhuma
inferência é feita sobre ele e ele não é salvo.

Uso:
    python preparar_dados.py <repo_secfednet> <CICIDS2017_improved.zip> <checkpoint_cnn1d.pt> <saida.npz>
"""

import argparse
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

TREINO_SHA256 = "b1a7e316661ebe6e0d5bb87ec712e989739a9768c38baad2096d91042a9d2247"
SPLIT_SHA256 = "5511442dcb01580d7cd327f312ca2c651e7ed194f4628f2246387de0bd779447"
ZIP_SHA256 = "97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883"
CLIENTES = 5
SEED_PARTICAO = 2305


def sha256(caminho):
    with open(caminho, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    p.add_argument("zip_dados", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("saida", type=Path)
    a = p.parse_args()

    base = a.repo / "secfednet-local" / "artigo-01"
    sys.path[:0] = [str(base / "cicids2017-corrigido"), str(a.repo / "integracao" / "artigo-01")]
    import piloto_fedavg as piloto
    from avaliar_corrigido import ler_sexta, padronizar
    from avaliar_piloto import distribuir_dirichlet

    split = base / "cicids2017-corrigido" / "split_cronologico.npz"
    assert sha256(a.zip_dados) == ZIP_SHA256, "ZIP do dataset diferente do usado pelo Aluno 1"
    assert sha256(split) == SPLIT_SHA256, "split_cronologico.npz diferente do Artigo 01"

    print("Lendo friday.csv...", flush=True)
    dados = ler_sexta(a.zip_dados)
    with np.load(split, allow_pickle=False) as m:
        assert dados.colunas == m["ordem_features"].tolist()
        treino, teste = [np.flatnonzero(np.isin(dados.linhas, m[k])) for k in ("linhas_treino", "linhas_teste")]
    treino, x, _, ajustes = padronizar(dados, treino, teste)
    hash_treino = hashlib.sha256(dados.linhas[treino].astype("<i8").tobytes()).hexdigest()
    assert hash_treino == TREINO_SHA256, "treino final não bate com o do Aluno 1"

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    bruto = dados.x[treino]
    media, escala = bruto.mean(axis=0), bruto.std(axis=0)
    escala[escala == 0.0] = 1.0
    assert np.array_equal(media, ck["media"].numpy()) and np.array_equal(escala, ck["escala"].numpy())
    assert ck["contrato"]["ordem_features"] == dados.colunas

    y = dados.y[treino]
    with patch.object(piloto, "CLIENTES", CLIENTES):
        particoes, tentativa = distribuir_dirichlet(y, SEED_PARTICAO)
    cliente = np.full(len(y), -1, dtype=np.int64)
    for i, idx in enumerate(particoes):
        cliente[idx] = i
    assert (cliente >= 0).all()

    np.savez_compressed(a.saida, x=x.astype(np.float32), bruto=bruto, y=y, cliente=cliente,
                        linhas=dados.linhas[treino], colunas=np.array(dados.colunas),
                        media=media, escala=escala)
    print(f"Treino final reproduzido: {len(y)} fluxos (hash confere), média/escala iguais às do checkpoint.")
    print(f"Expurgos: {ajustes}")
    for i in range(CLIENTES):
        m = cliente == i
        print(f"  cliente {i}: {m.sum():>6} fluxos | BENIGN {int((y[m]==0).sum()):>6} | DDoS {int((y[m]==1).sum()):>6}")


if __name__ == "__main__":
    main()
