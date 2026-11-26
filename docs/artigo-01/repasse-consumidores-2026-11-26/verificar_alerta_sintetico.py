"""Verifica a fixture de alerta e suas restrições de dados sintéticos."""
import argparse
from datetime import datetime
import ipaddress
import json
import math
from pathlib import Path


def exigir(ok, mensagem):
    if not ok:
        raise ValueError(mensagem)


def inteiro(valor, maximo):
    return type(valor) is int and 0 <= valor <= maximo


def verificar(evento):
    exigir(set(evento) == {"versao", "sintetico", "alerta_id", "fluxo", "classificacao", "modelo"}, "campos")
    exigir(evento["versao"] == "secfednet-alerta-proposta-v0" and evento["sintetico"] is True, "marcação sintética")
    exigir(evento["alerta_id"].startswith("fixture:"), "ID do alerta")
    fluxo = evento["fluxo"]
    exigir(set(fluxo) == {"id", "observado_em", "origem", "destino", "protocolo"}, "campos do fluxo")
    exigir(fluxo["id"].startswith("fixture:") and inteiro(fluxo["protocolo"], 255), "ID ou protocolo")
    tempo = fluxo["observado_em"]
    exigir(tempo.endswith("Z") and "T" in tempo and datetime.fromisoformat(tempo).tzinfo is not None, "tempo UTC")
    reservadas = [ipaddress.ip_network(x) for x in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
    for lado in ("origem", "destino"):
        endpoint = fluxo[lado]
        exigir(set(endpoint) == {"ip", "porta"} and inteiro(endpoint["porta"], 65535), "endpoint")
        ip = ipaddress.IPv4Address(endpoint["ip"])
        exigir(any(ip in rede for rede in reservadas), "IP deve pertencer a rede de documentação")
    c = evento["classificacao"]
    exigir(set(c) == {"taxonomia", "rotulo", "score", "semantica_score", "limiar"}, "campos da classe")
    exigir(c["taxonomia"] == "taxonomia-ficticia-v0" and isinstance(c["rotulo"], str) and bool(c["rotulo"]),
           "taxonomia ou rótulo")
    exigir(c["semantica_score"] == "valor_ilustrativo_sem_modelo" and c["limiar"] is None, "semântica ou limiar")
    exigir(type(c["score"]) in (int, float) and math.isfinite(c["score"]) and 0 <= c["score"] <= 1, "score")
    exigir(evento["modelo"] == {"versao": "fixture-sem-modelo", "checkpoint_sha256": None}, "modelo fictício")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arquivo", type=Path)
    a = p.parse_args()
    verificar(json.loads(a.arquivo.read_text(encoding="utf-8")))
    print("OK: fixture sintética, fluxo de documentação, sem modelo ou limiar operacional")
