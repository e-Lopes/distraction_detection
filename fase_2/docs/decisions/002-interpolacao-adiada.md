# 002 — Interpolação adiada

## Decisão

O primeiro pipeline usará zero-fill como baseline, acompanhado de `face_detected`. A
interpolação de gaps não será implementada nem usada nos primeiros treinamentos.

## Consequências

- Valores ausentes permanecem distinguíveis de medições válidas por uma flag explícita.
- O código não deve transformar zeros da extração em falhas sem conhecer sua proveniência.
- Interpolação permanece experimento futuro e não bloqueia H3.
