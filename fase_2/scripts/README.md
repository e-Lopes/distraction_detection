# Scripts de execução e manutenção

Execute na raiz do repositório, usando o mesmo ambiente Python da interface.

1. `python -m fase_2.scripts.audit_experiment_readiness`: auditar a prontidão experimental.
2. Scripts `g48a_*` e `run_g48a_all_frames.sh`: extração e validação dos frameworks faciais.
3. `run_measurement_experiment.sh` e `run_modern_experiment.sh`: preparar, treinar e relatar
   os novos experimentos em ambiente Bash/CUDA. A interface oferece os mesmos perfis.
4. `run_g2.ps1`: execução histórica G2 em PowerShell.
5. `python -m fase_2 results`: reconstruir a galeria local de relatórios e figuras.

`setup_g47_env.sh` prepara o ambiente específico de G4.7. As etapas e os requisitos
científicos estão em [protocolos](../docs/README.md); a presença de um script não
significa que seus dados estejam prontos para execução.
