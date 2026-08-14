from fase_2.src.training.g3_report import delta_statistics, select_representation


def test_delta_statistics_preserves_paired_fold_unit():
    rows=[{"model":"tcn","subset":"validation","comparison":"R2-R0","metric":"macro_f1_all_classes","delta":v} for v in (0.1,-0.1,0.2,0.0)]
    result=delta_statistics(rows)[0]
    assert result["n_folds"]==4
    assert abs(result["mean_delta"]-0.05)<1e-12


def test_selection_prefers_macro_f1_then_lower_variability():
    rows=[]
    for rep,mean,std in (("R0",0.4,0.02),("R1",0.3,0.01),("R2",0.4,0.03)):
        for model in ("lstm","tcn","transformer"):
            rows.append({"model":model,"representation":rep,"subset":"validation","metric":"macro_f1_all_classes","mean":mean,"standard_deviation":std})
    selected,ranking=select_representation(rows)
    assert selected=="R0"
    assert ranking[0]["representation"]=="R0"
