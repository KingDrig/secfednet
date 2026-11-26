# Baselines para Leonam · Artigo 03

O controle experimental disponível para pareamento com a avaliação de Pedro é o ensaio limpo **FedAvg-SGD × SCAFFOLD, 20 rodadas, BENIGN × DDoS**. O dossiê da expansão também tem uma logística e CNN de integração, com alvo **BENIGN × ATTACK** e outro split. Identifique no protocolo qual baseline será usada.

| Registro | Caminho no repositório ou no acervo local | Versão |
| --- | --- | --- |
| Treino FedAvg-SGD/SCAFFOLD e wrapper | `secfednet-local/artigo-01/cnn-scaffold/{experimentar_20.py,experimentar_scaffold.py,scaffold.py,reproduzir_20.sh}` | Base `0b38f25`; evidências de reprodução em `d27c89de` |
| Protocolo e configuração efetiva | `secfednet-local/artigo-01/cnn-scaffold/protocolo_20.json` e `ensaio-20-rodadas/execucao/configuracao.json` | `secfednet-scaffold-20rodadas-v1` |
| Split histórico | `secfednet-local/artigo-01/cnn-validacao/particao/manifesto_validacao.npz` | SHA `bb710d6492ac83456b61ba8469d06f6729bf063ba586a557b0165a701173339f` |
| Posições de treino dos cinco clientes | `secfednet-local/artigo-01/cnn-scaffold/ensaio-20-rodadas/execucao/clientes.npz` | SHA `bbb35ecd842ddf7924e6b9d884d096a36bac01a258affaee27af34b592ddaffb` |
| Ambiente histórico | `secfednet-local/artigo-01/cnn-flower/requirements.lock` | Linux x86-64, Python 3.14.4, NumPy 2.3.5, Torch 2.12.1+cpu, Flower 1.32.0 |

Ajuste 203.251 e validação interna 50.815, sem interseção de linhas. A posição local do cliente aponta para `manifesto["ajuste"][posição]`. Suportes totais por cliente 4.605 / 19.982 / 6.300 / 50.749 / 121.615, com cobertura exata e única. Cinco clientes em toda rodada, SGD lr 0,01 sem momentum ou decay, lote 512, uma época local, agregação uniforme, seed do modelo 23, partição Dirichlet por classe α=0,5 seed 2305. As ordens efetivas e respectivos hashes constam em `configuracao.json`. O Artigo 02 tem outras cinco fatias, criadas retrospectivamente somente para avaliação. Macro-F1 global histórico 0,9849740690 / 0,9003566947 respectivamente. Adoção de uma partição candidata e de uma definição de FF1 requer confirmação.

Da raiz do acervo:

```bash
a=secfednet-local/artigo-01
sha256sum -c "$a/cnn-scaffold/SHA256_FONTES_20"
sha256sum -c "$a/cnn-scaffold/SHA256_ARTEFATOS_20"
# Comando histórico de treino, para futura reprodução em destino novo.
bash "$a/cnn-scaffold/reproduzir_20.sh" /tmp/secfednet-baseline20-nova
```

O wrapper executa 20 rodadas em cada método e usa referências do ensaio curto e os dados corrigidos. A conferência documental independente pode ser feita em checkout de `cf8b332` ou `d27c89de` com `python3 -I -B docs/reproducao-vm/2026-11-24/verificar.py`. Esta missão verificou as partições, os hashes dos artefatos referenciados e as métricas, sem repetir o treino.

## Baseline logística da expansão

Em `cf8b332`, a pasta `docs/artigo-01/expansao-2026-11-25/fonte/` tem `baseline.py`, `baseline_config.json`, `verificar_candidato.py`, `reproduzir.sh` e o lock. Linear 81→1, BCE, Adam lr 0,001, três épocas, lote 4096, seed 23 e limiar 0,5. Alvo 0=BENIGN, 1=ATTACK. Segunda a quarta treino, quinta validação, sexta reservada. Máscaras e índices principais/sensibilidade em `manifesto_indices_candidato.npz`, SHA `af1c057c2ca9114c4bcad8c2da5b8ce265c2d22427f7771fd8ae983cba17bdd6`. Partição de cinco clientes da CNN de integração, separada da logística centralizada, disponível no acervo `fedavg_integracao/clientes_treino_idx0.npz` da execução 20h46. Veja os hashes no inventário de artefatos.

Com a pasta do commit extraída e o acervo acessível:

```bash
repo=/caminho/para/secfednet
py="$repo/secfednet-local/artigo-01/cnn-flower/.venv/bin/python"
orig="$repo/secfednet-local/artigo-01/execucoes/2026-11-25/protocolo-candidato-validacao-20-46-58"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
e=docs/artigo-01/expansao-2026-11-25
"$py" -I -B "$e/fonte/baseline.py" testar
"$py" -I -B "$e/fonte/verificar_candidato.py" "$orig/candidato"
tmp=$(mktemp -d)
cp -a "$orig/baseline" "$tmp/baseline"
"$py" -I -B "$e/fonte/baseline.py" verificar "$orig/candidato" "$tmp/baseline"
```

Esses comandos passaram. As 12 matrizes e as máscaras foram verificadas. A recarga CPU `weights_only=True` reproduziu bit a bit os scores da quinta e as métricas. A verificação escreveu seu JSON em cópia temporária. O teste reservado foi lido somente para as checagens de representação e partição. Para reprodução futura, `bash "$e/fonte/reproduzir.sh" NOVO_DESTINO "$py" CAMINHO_DO_ZIP` a partir do checkout com o dossiê.

## Compatibilidade e decisões

Opacus não estava instalado no ambiente conferido. Permanecem pendentes a validação do módulo no ambiente de Leonam, as transformações do carregador e a contabilidade de privacidade por cliente com as mesmas partições. Selecionar e documentar a baseline limpa, como contabilizar diferenças de lotes/otimizador e o protocolo de avaliação antes da comparação. Os resultados de defesa da equipe dependem do cenário e da referência do Artigo 02. O histórico Linux/CPU não estabelece compatibilidade em Windows nativo nem no ambiente de Leonam.
