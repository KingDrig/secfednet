"""Valida ou atualiza a tabela CICIDS do README a partir do CSV do manuscrito."""
import argparse
import csv
from pathlib import Path
import re

COLUNAS = (
    "bruto", "excl_attempted", "excl_nao_finito", "excl_grupo", "excl_vetor_bruto",
    "excl_vetor_modelo", "treino", "validacao", "teste_reservado",
)
PADRAO = re.compile(
    r"(^\| Rótulo CICIDS2017 corrigido \|[^\n]*\n\| ---[^\n]*\n)"
    r"((?:\|[^\n]*\n)+)", re.MULTILINE
)


def tabela(path):
    with path.open(encoding="utf-8", newline="") as f:
        linhas = [r for r in csv.DictReader(f) if r["base"] == "CICIDS2017 corrigido"]
    totais = [r for r in linhas if r["rotulo"] == "TOTAL"]
    if len(totais) != 1:
        raise ValueError("O CSV deve conter um único TOTAL do CICIDS2017")
    classes = [r for r in linhas if r["rotulo"] != "TOTAL"]
    tentativas = [r for r in classes if r["rotulo"].endswith(" - Attempted")]
    nominais = [r for r in classes if not r["rotulo"].endswith(" - Attempted")]
    if len(tentativas) != 11 or any(int(r["excl_attempted"]) for r in nominais):
        raise ValueError("São esperadas 11 variantes Attempted e exclusões separadas das classes nominais")
    for r in classes:
        if int(r["bruto"]) != sum(int(r[k]) for k in COLUNAS[1:]):
            raise ValueError(f"Contagem não conciliada: {r['rotulo']}")
    agregada = {"rotulo": f"{len(tentativas)} rótulos Attempted",
                **{k: sum(int(r[k]) for r in tentativas) for k in COLUNAS}}
    if [agregada[k] for k in COLUNAS] != [11979, 11979] + [0] * 7:
        raise ValueError("Agregado Attempted diverge do inventário auditado")
    if any(int(totais[0][k]) != sum(int(r[k]) for r in classes) for k in COLUNAS):
        raise ValueError("TOTAL diverge da soma das classes")
    return "".join(
        "| " + r["rotulo"] + " | "
        + " | ".join(f"{int(r[k]):,}".replace(",", ".") for k in COLUNAS) + " |\n"
        for r in nominais + [agregada] + totais
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gravar", action="store_true", help="Atualiza somente a tabela no README")
    p.add_argument("--pasta", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    caminho = a.pasta / "README.md"
    original = caminho.read_text(encoding="utf-8")
    corpo = tabela(a.pasta / "tabela_manuscrito.csv")
    novo, n = PADRAO.subn(lambda m: m[1] + corpo, original)
    if n != 1:
        p.error("A tabela CICIDS deve aparecer uma única vez no README")
    if a.gravar and novo != original:
        caminho.write_text(novo, encoding="utf-8")
    elif novo != original:
        p.error("A tabela do README diverge do CSV")
    print(next(linha for linha in corpo.splitlines() if "11 rótulos Attempted" in linha))


if __name__ == "__main__":
    main()
