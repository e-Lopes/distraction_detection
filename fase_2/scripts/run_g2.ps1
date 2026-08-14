$ErrorActionPreference = "Stop"

$configs = @(
    @{ window = 30; path = "fase_2/configs/experiment/g2_temporal_r0_w30.yaml" },
    @{ window = 60; path = "fase_2/configs/experiment/g2_temporal_r0_w60.yaml" },
    @{ window = 150; path = "fase_2/configs/experiment/g2_temporal_r0_w150.yaml" }
)

$logDirectory = "fase_2/outputs/logs/G2/orchestrator_b1024"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

foreach ($item in $configs) {
    $window = $item.window
    $logPath = Join-Path $logDirectory "window_$window.log"
    "START window=$window timestamp=$([DateTime]::UtcNow.ToString('o'))" | Add-Content $logPath
    python -m fase_2.src.training.temporal_multiseed `
        --experiment-config $item.path `
        --checkpoint-dir fase_2/outputs/models/G2 `
        --run-dir fase_2/outputs/logs/G2 `
        --prediction-dir fase_2/outputs/predictions/G2 `
        --output-dir fase_2/outputs/metrics/G2 `
        --figure-dir "fase_2/outputs/figures/G2/w$window" `
        --max-runs 12 *>> $logPath
    if ($LASTEXITCODE -ne 0) {
        "FAILED window=$window exit=$LASTEXITCODE" | Add-Content $logPath
        exit $LASTEXITCODE
    }
    "COMPLETE window=$window timestamp=$([DateTime]::UtcNow.ToString('o'))" | Add-Content $logPath
}
