# A'Space adapter — tabfm

**Adresse :** `fiber:a0:tabfm` — D-N053 pas A (design Clara, dispatch D-N053).
**Fork souverain :** `Amdkn/tabfm` (parent `google-research/tabfm`, branche `main`).
**Gel amont :** `fbb665569425fd2f490c6576b3af967876fe11ff` (« Merge pull request #97 from asdespot/fix/pytorch-jaxtyping-array-import »).
**Bind Cordis Fiber (16L) :** `a8d6be922db6b617667d9caabec583f8acb6df4d` — `aspace-plugin.json`, re-applique VERBATIM sur `main` au-dessus de l'adapter (F8 D2).
**Licence :** `Apache-2.0`.
**Version adapter :** 0.1.0 — contrat seul, patron D-N048 (dify / cordis).

## Ce que fait l'adapter

TabFM : Tabular Foundation Model, classification/regression zero-shot par in-context learning (backends jax / pytorch). Poids non-commerciaux, confines au profil interne.

Surface : `validate()` (constate : gel, licence, importabilite amont, poids
presents dans le fork, zero appel reseau), `doctor()` (liste de controles),
`describe()` (manifeste machine). **Aucun fichier amont touche, aucun appel
reseau, aucun poids monte dans le fork.**

## Confinement

CONFINEMENT PROFIL INTERNE (D-N052 D3, design D-N053 D3) : les POIDS TabFM sont non-commerciaux ; jamais en B2B, jamais montes dans le fork (les miroirs HF sont le travail R1). Le code du repo est Apache-2.0.

## Tests

```bash
python -m pytest aspace-tabfm-adapter/tests/test_smoke.py -v
```

Smoke : constantes (gel 40-hex, adresse fiber, licence), `validate()` vert,
`doctor()` tout ok, import amont, zero import reseau dans l'adapter.
La suite amont est celle du module au gel (voir la fiche d'adaptation :
`tabfm-ADAPTATION.md`, evidence D-N053/a/tabfm/).
