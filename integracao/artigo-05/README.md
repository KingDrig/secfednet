# Artigo 05 · Explicabilidade com SHAP do SecFedNet

**Passos 1 e 2 concluídos; Passos 3 e 4 em ensaio.** A CNN do Artigo 01 decide principalmente pelo tamanho dos pacotes: essa família responde por 51% da importância, e 8 das 81 features explicam metade das decisões (Passo 1). Os 5 clientes federados olham para as mesmas features que o global; o que muda entre eles é a intensidade com que acusam DDoS (Passo 2). Trocar só as 4 features mais importantes do SHAP por valores de tráfego normal faz 97% dos DDoS passarem despercebidos; trocar as 8 menos importantes não engana nenhum (ensaio do Passo 3). O treino adversarial federado dificulta o ataque, mas não fecha o ponto cego: o modelo passa a depender de outras features manipuláveis (ensaio do Passo 4).

Título de trabalho: *Opening the Black Box: SHAP-Based Explainability of Adversarial Blind Spots in SecFedNet*.

## Entradas (fora do Git)

Os pacotes e o dataset circulam pelo canal autorizado do Willian, não pelo Git. Confira cada um antes de rodar.

| Entrada | SHA-256 | Origem |
| --- | --- | --- |
| `CICIDS2017_improved.zip` | `97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883` | CNS2022, o mesmo do Artigo 01 |
| `secfednet-cnn1d-v1.zip` | `e7d042b38c77b96bef54533302bef817d6b8a4a8e55b717bac7b32f98b5b6014` | Pacote de inferência do Artigo 01 |
| `checkpoint_cnn1d.pt` (dentro do ZIP acima) | `37db7d4f6d0123786cc4ed5227d670edb1344c07089e8ff6910c6446e57400f3` | FedAvg, 3 rodadas |
| `secfednet-shap-fedavg3.zip` | `9eac2d389dbbc6e7e4eec030507b896fe51be4f634d0f6c0b8903a079b5bab92` | Global e 5 modelos locais da rodada 3, com os índices da partição (Passo 2) |

Do próprio repositório são usados, sem cópia: `secfednet-local/artigo-01/cicids2017-corrigido/` (`ler_sexta`, `padronizar`, `split_cronologico.npz`), `integracao/artigo-01/avaliar_piloto.py` (`distribuir_dirichlet`) e `secfednet-local/artigo-01/cnn-flower/cnn1d.py`. O script confere o SHA-256 de cada um.

## Como rodar

```bash
pip install -r requirements.txt
python preparar_dados.py <repo_secfednet> <CICIDS2017_improved.zip> <checkpoint_cnn1d.pt> dados_treino.npz
python shap_global.py <repo_secfednet> dados_treino.npz <checkpoint_cnn1d.pt> resultados
python shap_clientes.py <repo_secfednet> dados_treino.npz <pasta_extraida_secfednet-shap-fedavg3> resultados
python evasao_ensaio.py <repo_secfednet> dados_treino.npz <pasta_extraida_secfednet-shap-fedavg3> resultados
python defesa_ensaio.py <repo_secfednet> dados_treino.npz <pasta_extraida_secfednet-shap-fedavg3> resultados
sha256sum -c SHA256SUMS
```

- `preparar_dados.py` (1 a 2 minutos) reconstrói o treino final do Artigo 01 e a partição dos 5 clientes. Ele aborta se o hash do treino (`b1a7e316…a9d2247`) ou a média e a escala do checkpoint não baterem.
- `shap_global.py` (alguns minutos em CPU) gera `resultados/`. A execução é determinística: rodar de novo reproduz os mesmos CSV e JSON.
- `dados_treino.npz` é gerado localmente e não é versionado (`.gitignore`).

## Decisões de método (para validação do Artigo 01)

A matriz de dependências do repasse de 26/11 pede validar estes três pontos.

| Ponto | Escolha | Motivo |
| --- | --- | --- |
| Alvo da explicação | Logit de DDoS (antes da sigmoide), via wrapper que descarta o embedding | O SHAP soma linearmente no logit; na sigmoide, a saturação perto de 0 e 1 esconde as contribuições |
| Explainer | `shap.DeepExplainer` (DeepLIFT), com `GradientExplainer` como segunda opinião | Rede pequena (1.149 parâmetros); aditividade exata verificada |
| Fundo e amostra | 500 fluxos BENIGN de fundo; 2.000 explicados (1.000 DDoS e 1.000 BENIGN), sorteados do treino final | O teste de sexta permanece reservado. Explicar o treino descreve como o modelo decide, não o desempenho em dados novos |

As features entram normalizadas como na CNN (média e desvio do treino), e os valores SHAP estão em unidades de logit.

## Resultados

| Família | % da importância | Features |
| --- | ---: | ---: |
| Tamanho de pacote | 51,1% | 17 |
| Tempo entre pacotes (IAT) | 10,2% | 14 |
| Volume (bytes) | 10,1% | 4 |
| Flags TCP | 9,7% | 14 |
| Atividade e ociosidade | 5,5% | 8 |
| Taxas por segundo | 5,1% | 5 |
| Bulk | 5,1% | 6 |
| Cabeçalho e janela TCP | 1,6% | 4 |
| Contagem de pacotes | 0,9% | 5 |
| Duração do fluxo | 0,6% | 2 |
| ICMP | 0,0% | 2 |

Top 4 features: `Bwd Segment Size Avg`, `Packet Length Variance`, `Average Packet Size` e `Subflow Bwd Bytes`. Seis features têm importância zero: `Fwd URG Flags`, `Bwd URG Flags`, `URG Flag Count`, `Subflow Bwd Packets`, `ICMP Code` e `ICMP Type`.

| Teste de confiança | Resultado |
| --- | --- |
| Aditividade (soma SHAP + base = logit) | erro máximo 1,1 × 10⁻⁴ |
| Estabilidade entre 3 sementes | Spearman 0,94; 9 do top-10 em comum |
| DeepSHAP × GradientSHAP | Spearman 0,998; 9 do top-10 em comum |
| DeepSHAP × importância por permutação | Spearman 0,58; top-4 iguais; 5 do top-10 em comum |

A concordância menor com a permutação se explica pelas features quase duplicadas (por exemplo `Packet Length Std` e `Packet Length Variance`): embaralhar uma delas pouco afeta o modelo enquanto a gêmea permanece.

## Passo 2: SHAP por cliente

O global e os 5 modelos locais (rodada 3, depois do treino local e antes da agregação) foram explicados sobre **os mesmos 2.000 fluxos e o mesmo fundo do Passo 1**, de modo que as diferenças vêm só dos modelos. O `shap_clientes.py` confere os hashes dos 6 modelos e verifica que os índices do pacote coincidem com a partição reconstruída.

| Modelo | DDoS no treino do cliente | Logit médio nos DDoS | Recall DDoS (treino completo) | Falso alarme |
| --- | ---: | ---: | ---: | ---: |
| Global | 50.194 | 4,88 | 99,96% | 0,02% |
| Cliente 0 | 26.999 | 4,45 | 100,00% | 0,77% |
| Cliente 1 | 735 | 3,18 | 99,77% | 0,02% |
| Cliente 2 | **0** | **0,26** | **68,69%** | 0,00% |
| Cliente 3 | 457 | 2,75 | 98,99% | 0,01% |
| Cliente 4 | 22.003 | 6,49 | 99,98% | 0,01% |

1. **Todos olham para as mesmas features.** O ranking dos modelos concorda com Spearman de 0,90 a 1,00 e 8 a 10 do top-10 em comum. O tamanho de pacote responde por 48% a 52% da importância em todos. Nenhum cliente desenvolveu uma "visão" própria.
2. **A diferença está na intensidade.** O cliente 2 nunca viu DDoS: depois do treino local, ele continua olhando para as mesmas features, mas as usa com bem menos força. Nos fluxos DDoS, `Bwd Segment Size Avg`, `Packet Length Variance` e `Average Packet Size` perdem 2,0, 1,7 e 1,1 de logit em relação ao global, o logit médio cai para 0,26 (quase no limiar) e o recall vai a 68,7%. Clientes com muito DDoS (4) acusam com mais força; o cliente 0, com 83% de DDoS, também gera mais falsos alarmes.
3. **O global é a média das explicações locais.** A explicação do global coincide com a média ponderada das explicações locais (Spearman 0,999), e nenhuma feature importante para um cliente aparece diluída no global.
4. **O global trata os dados de cada cliente do mesmo jeito.** Explicado sobre amostras de cada cliente, o ranking do global se mantém (Spearman 0,92 a 1,00). O menor valor é o do cliente 2, que só tem tráfego BENIGN.

**Leitura para o artigo:** o ponto cego do tamanho de pacote é **coletivo**: não vem de um cliente específico, e todos os modelos da federação compartilham a dependência dessas features. Um atacante que manipule o tamanho dos pacotes afeta todos os participantes.

**Ressalva de método:** no FedAvg, cada modelo local parte do global da rodada anterior e treina uma única época. Parte da semelhança entre os modelos vem desse ponto de partida comum. Para medir o que cada cliente aprenderia por conta própria, o experimento complementar é treinar um modelo só com os dados de cada cliente, a partir do zero.

## Passo 3 (ensaio): evasão guiada pelo SHAP e mapa de blind spots

**Natureza do ensaio.** É um ataque no espaço de features, não tráfego realizável: em 2.000 fluxos DDoS que o global detecta (do treino final, fora da amostra e do fundo dos Passos 1 e 2), os valores de um grupo de features são trocados pelos de um fluxo BENIGN sorteado. Serve para testar se o ranking do SHAP prevê onde o modelo é vulnerável. A versão final depende da definição dos dados de evasão e dos exemplos da GAN do Artigo 04, que respeitam as restrições do tráfego real.

| Features trocadas | Nº | % da importância SHAP | Evasão no global | Evasão média nos 5 locais |
| --- | ---: | ---: | ---: | ---: |
| Top-4 SHAP | 4 | 39% | 96,7% | 97,0% |
| Top-8 SHAP | 8 | 53% | 97,7% | 97,0% |
| Tamanho de pacote (família inteira) | 17 | 51% | 98,0% | 98,0% |
| Tamanho de pacote + Volume | 21 | 61% | 98,1% | 98,3% |
| Duração + Contagem de pacotes | 7 | 1,5% | 0,6% | 8,8% |
| Flags TCP | 14 | 9,8% | 1,4% | 13,4% |
| 8 menos importantes | 8 | 0,0% | 0,0% | 6,3% |
| 8 sorteadas (média de 20 sorteios) | 8 | — | 7,3% | 21,7% |
| Todas as 81 (controle) | 81 | 100% | 100% | 99,8% |

A média dos locais nos controles inclui o cliente 2, que já deixa passar 30% dos DDoS sem ataque nenhum.

1. **O SHAP prevê a vulnerabilidade.** Mexer em 4 features (5% das 81) que concentram 39% da importância engana quase todo o detector. Mexer nas features que o SHAP diz serem irrelevantes não engana nada.
2. **O ataque vale para a federação inteira.** Global e locais caem juntos, coerente com o ponto cego coletivo do Passo 2.
3. **O atacante precisa mover bastante, mas não tudo.** Na curva de orçamento, mover as top-4 features 50% do caminho até o tráfego normal engana 35% dos DDoS; 70% do caminho engana 88%.

**Mapa de blind spots** (depois do ataque Top-8, nos 1.954 fluxos que enganaram o global): 12 features ainda separam o ataque disfarçado do tráfego normal com nitidez (separação ≥ 0,8, de 0 a 1). Quatro delas têm importância SHAP abaixo da mediana, ou seja, denunciariam o ataque, mas o modelo quase não as usa:

| Feature | Família | Posição no SHAP | Separação depois do ataque |
| --- | --- | ---: | ---: |
| `Bwd Packet Length Std` | Tamanho de pacote | 45ª | 1,00 |
| `Total Length of Fwd Packet` | Volume (bytes) | 49ª | 0,91 |
| `Fwd RST Flags` | Flags TCP | 52ª | 0,89 |
| `Total Length of Bwd Packet` | Volume (bytes) | 74ª | 0,86 |

Várias das features que continuam separando são parentes físicas das alteradas. Por exemplo, `Bwd Packet Length Std` tem correlação 0,97 com `Packet Length Variance`, que o atacante mudou. Há duas leituras para o artigo: (a) **defesa**: checar a coerência entre features relacionadas pegaria um ataque ingênuo; (b) **limitação**: um atacante realista ajustaria as parentes também, e por isso o mapa final precisa dos exemplos com restrições do Artigo 04. Mesmo trocando a família de tamanho inteira (17 features, mais coerente), a evasão é de 98%.

## Passo 4 (ensaio): corrigir o ponto cego

Partindo do global do Artigo 01, foram feitas mais 3 rodadas de FedAvg nos mesmos 5 clientes, com os hiperparâmetros do Artigo 01 (Adam lr = 0,001 reiniciado a cada rodada, 1 época local, lote 512), em três variantes:

- **Controle:** treino extra sem nada novo, para separar o efeito de "treinar mais".
- **Adversarial top-8:** cada cliente acrescenta cópias disfarçadas dos próprios DDoS, com as top-8 features do SHAP trocadas por valores BENIGN do próprio cliente.
- **Adversarial família:** o mesmo, trocando as 21 features de tamanho de pacote e volume.

O cliente 2 não tem DDoS e treina sem cópias. 10% do treino (25.407 fluxos) ficam reservados para a avaliação e não entram no treino extra. O global original já tinha visto esses fluxos, então o desempenho limpo dele é otimista. O treino FedAvg é simulado em processo, sem Flower.

| Modelo | Recall DDoS | Falsos alarmes | Top-4 SHAP | Top-8 SHAP | Tamanho (17) | Tamanho + Volume (21) | Ataque adaptativo: features para 90% de evasão |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 99,92% | 4 (0,02%) | 97% | 98% | 98% | 98% | 2 |
| Controle | 99,96% | 1 (0,005%) | 97% | 98% | 98% | 98% | 2 |
| Adversarial top-8 | 100% | 7 (0,03%) | 45% | **4%** | 97% | 97% | 12 |
| Adversarial família | 100% | 68 (0,33%) | 16% | 20% | **41%** | **39%** | 12 |

As colunas de ataque mostram a evasão: a % dos 2.000 DDoS da avaliação que passam como BENIGN. No **ataque adaptativo**, o atacante refaz o SHAP no modelo corrigido e troca as top-k features desse modelo.

1. **Treinar mais não ajuda.** O controle fica igual ao original; o efeito vem dos exemplos disfarçados.
2. **A correção funciona contra o ataque que ela viu, e pouco além dele.** O treino com as top-8 derruba esse ataque de 98% para 4%, mas não protege contra a troca da família inteira (97%). Treinar com a família inteira protege mais (41%), ao custo de 17 vezes mais falsos alarmes (0,33%).
3. **O ponto cego muda de lugar, não desaparece.** Um atacante que refaz o SHAP precisa trocar 12 features em vez de 2 para chegar a 90% de evasão: o custo do ataque sobe 6 vezes, mas o ataque continua possível. Depois da correção por família, o modelo passa a depender do tempo entre pacotes (de 10% para 29% da importância) e menos do volume (de 10% para 2%). Ou seja, ele troca uma dependência manipulável por outra.

**Leitura para o artigo:** o SHAP serve dos dois lados. Ele mostra ao defensor onde reforçar e ao atacante para onde ir. Uma defesa duradoura precisa de features que o atacante não controla, ou de restrições de coerência entre features. Saber quantas das 12 features um atacante real consegue alterar de forma coerente depende dos exemplos com restrições do Artigo 04. Onde a correção entra (aqui ou no Artigo 03, de defesa) fica para combinar.

## Arquivos

| Caminho | Conteúdo |
| --- | --- |
| `preparar_dados.py` | Reconstrução auditada do treino e da partição |
| `shap_global.py` | DeepSHAP, testes de confiança e figuras (Passo 1) |
| `shap_clientes.py` | SHAP dos 6 modelos sobre os mesmos fluxos e do global sobre os dados de cada cliente (Passo 2) |
| `resultados/clientes/` | Importância e contribuição por modelo, famílias por modelo, `divergencia.json` e 3 figuras |
| `evasao_ensaio.py` | Ensaio de evasão por transplante, curva de orçamento e mapa de blind spots (Passo 3) |
| `resultados/evasao/` | `taxas_evasao.csv`, `curva_orcamento.csv`, `mapa_blind_spots.csv`, `resumo.json` e 3 figuras |
| `defesa_ensaio.py` | Treino FedAvg de correção (controle e adversarial), avaliação e ataque adaptativo (Passo 4) |
| `resultados/defesa/` | `resultado.json`, `evasao_cenarios.csv`, `curva_adaptativa.csv`, `familias_por_modelo.csv` e 2 figuras |
| `resultados/importancia_features.csv` | Ranking das 81 features: DeepSHAP geral e por classe, GradientSHAP, permutação |
| `resultados/importancia_familias.csv` | Importância agregada por família |
| `resultados/confianca.json` | Aditividade, estabilidade e concordância entre métodos |
| `resultados/figuras/` | `top20_features`, `familias`, `beeswarm_ddos`, `deep_vs_permutacao` |
| `PROVENIENCIA.json` | Commits, hashes de entrada, ambiente e parâmetros da execução de referência |
| `SHA256SUMS` | Manifesto dos arquivos desta pasta |

## Próximos passos

1. **Passo 3 final:** repetir o ensaio nos dados que o Willian definir para a evasão e cruzar o mapa de blind spots com os exemplos da GAN do Artigo 04.
2. **Passo 4 final:** repetir com os dados de evasão definidos e os exemplos do Artigo 04; testar uma defesa por coerência entre features relacionadas; combinar com o Artigo 03 onde entra a correção.
3. **Complementar ao Passo 2:** treinar um modelo do zero só com os dados de cada cliente, para medir o que cada um aprenderia sem a federação.
