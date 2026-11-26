# Matriz de dependências · Artigos 02 a 09

As designações vêm do Controle Artigos e da página 4 do roteiro de 25/11. O PDF de Pedro documenta o repasse preparado. A publicação anterior da baseline e do dossiê tem registro local, separado de recebimento pelos responsáveis. “Verificado” se refere ao artefato e ao teste indicado. [CSV com caminhos, versões, SHA-256 e comandos](MATRIZ_DEPENDENCIAS.csv). [Procedência](PROVENIENCIA_REVISAO.json) · [Canais de circulação](ACERVO_CANAIS.csv).

| Artigo | Responsável | Dependência | Estado | Evidência de circulação | Próxima ação |
| --- | --- | --- | --- | --- | --- |
| 02 | Pedro Nonato | Código do ensaio limpo e evidências globais de reprodução | verificado | baseline 0b38f25 com push histórico; d27 herdado de cf8; PDF de repasse preparado | usar os globais e limiares históricos como referência |
| 02 | Pedro Nonato | Avaliação por cliente com Macro-F1 binário | verificado | local; recebimento não verificado | revisar tabela e os dez resultados individuais |
| 02 | Pedro Nonato / Willian | Adoção da partição retrospectiva e da métrica por cliente | aguarda confirmação da equipe | local; recebimento não verificado | confirmar uso da candidata e definição FF1 antes de comparação; pesos locais 20 não localizados |
| 03 | Leonam Tsutsumi | Baseline limpa, posições dos cinco clientes e partições | verificado | fontes e relatório no Git; matrizes derivadas e pesos locais no acervo | escolher comparador limpo e manter as mesmas partições do ensaio escolhido |
| 03 | Leonam Tsutsumi | Compatibilidade Opacus e contabilidade de privacidade | falta produzir | sem artefato nesta conferência | validar versão suportada e preservar orçamento e validação |
| 03 | Leonam Tsutsumi / Pedro Nonato | Referência experimental do Artigo 02 para comparação em 03 | depende de outro artigo | sem artefato nesta conferência | confirmar referência de Pedro após revisão |
| 04 | Marina Alice / Juliany Leticia | CICIDS corrigido e recorte BENIGN reprodutível | verificado | fonte externa, cópia local; dossiê de cobertura no acervo | usar inventário/partições autorizadas; dataset circula por fonte separada |
| 04 | Marina Alice / Juliany Leticia | Detector multiclasse de tipos conhecidos | falta produzir | sem artefato nesta conferência | obter episódios independentes com mesmos tipos; hoje só BENIGN compartilha suporte em treino/validação |
| 04 | Marina Alice / Juliany Leticia | Checkpoint do detector de referência defendido | depende de outro artigo | sem artefato nesta conferência | aguardar versão de Leonam |
| 05 | Kaim Chavante | Detector, checkpoints por cliente, partição e referência exata do dataset | verificado | local; recebimento não verificado | validar background de treino, alvo da explicação e explainer |
| 05 | Kaim Chavante | Protocolo SHAP e insumo da comparação do Artigo 04 | depende de outro artigo | sem artefato nesta conferência | explicação do piloto disponível como base; confirmar escopo da comparação |
| 06 | Mateus Vasconcelos / Marcos Gabriel | Arquitetura, pesos, embedding e Dirichlet | verificado | local; recebimento não verificado | testar adaptação de entrada e normalização no domínio de fraude |
| 06 | Mateus Vasconcelos / Marcos Gabriel / Willian | Controle de pré-treino com rótulos embaralhados | falta produzir | sem artefato nesta conferência | decidir permutação, manter validação verdadeira e comparar com baseline pareada |
| 07 | Mateus Vasconcelos / Marcos Gabriel | Schema com fixture sintética para ingestão | aguarda confirmação da equipe | local; fixture verificada | revisar metadados, taxonomia, score e contrato de transporte |
| 07 | Mateus Vasconcelos / Marcos Gabriel / Willian | Alertas reais multiclasse UNSW com identificação de fluxo | falta produzir | sem artefato nesta conferência | depende de insumos UNSW, modelo validado e especificação dos alertas |
| 07 | Mateus Vasconcelos / Marcos Gabriel / Kaim | Referências de explicabilidade para triagem | depende de outro artigo | sem artefato nesta conferência | alinhar taxonomia do alerta com o método de explicação |
| 08 | Marina Alice / Juliany Leticia | Corpus e artefatos autorizados do sistema de triagem do Artigo 07 | depende de outro artigo | sem artefato nesta conferência | aguardar corpus e sistema de triagem. Fixture do Artigo 01 não substitui esse corpus |
| 09 | Willian Rodrigo / Gabriel Chaves | Hashes de modelos, dados, código e locks | verificado | evidências no histórico Git; ZIPs e dados externos continuam locais | usar hashes como entrada da auditoria de cadeia |
| 09 | Willian Rodrigo / Gabriel Chaves | SBOM e assinaturas por artefato | falta produzir | sem artefato nesta conferência | definir escopo, ferramentas, cadeia de confiança e chaves autorizadas |
| 01 | Gabriel Chaves | UNSW bruto, dicionário, ground truth e versões oficiais | falta produzir | evidência documental; insumos brutos ausentes | recuperar lista exata, auditar antes de treinar |
