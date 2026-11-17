# Guia rápido de leitura, validação e minimização

Este reference foi consolidado na skill v2.0.0. Não há aqui uma receita
específica de steering para thinking; a receita Concept Country e seus números
estão em `evidence-wsp-wsr.md`.

Use este mapa para não duplicar orientação:

1. **Fluxo operacional, contenção, controles e deploy:** veja o `SKILL.md` na
   raiz desta skill.
2. **Evidência e números de WSP/WSR, Qwen, Muse, iWSP e minimização:** veja
   `evidence-wsp-wsr.md`.
3. **Taxonomia de knowledge editing, merging, steering, LoRA e checklist
   estatístico completo:** veja `methods-and-validation.md`.
4. **Papers, repositórios e status de publicação:** veja `papers.md`.
5. **Pré-registro copiável, inclusive campos específicos de WSR e splits para
   minimização:** veja `experiment-template.md`.

Regras curtas:

- Reading prevê o efeito de um edit dado; não escolhe o edit nem substitui
  ground truth.
- Uma narrativa de pesos é hipótese até intervenção/comportamento confirmarem.
- Random norm-matched, no-op, dose-resposta, guards e teste final intocado são
  necessários; reter a base é válido.
- Minimize em discovery/selection e só depois congele os braços para o teste
  final. Se o teste já foi consumido, use um novo teste externo independente.
- Teste o artefato realmente servido. GGUF/quantização são condicionais ao
  destino; marque N/A fora deles.
- Ao citar um trabalho, ofereça o paper/PDF ou abra o link no browser mediante
  confirmação do usuário.
