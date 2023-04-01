from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from gptchem.evaluator import evaluate_classification
from gptchem.models.xgboost import XGBClassificationBaseline

WATER_STABILITY_FEATURES = features = [
    "M_AtomicRadii",
    "M_Affinity",
    "M_Ionization",
    "L_afpC2_C2_C3",
    "L_afpC2_C3_C3",
    "L_afpC3_C3_C3",
    "L_afpC3_C3_C4",
    "L_afpC3_C3_F1",
    "L_afpC3_C3_H1",
    "L_afpC3_C3_N2",
    "L_afpC3_C3_N3",
    "L_afpC3_C3_O1",
    "L_afpC3_C3_O2",
    "L_afpC3_C4_C3",
    "L_afpC3_C4_H1",
    "L_afpC3_C4_P4",
    "L_afpC3_N2_C3",
    "L_afpC3_N2_N2",
    "L_afpC3_N2_N3",
    "L_afpC3_N3_C3",
    "L_afpC3_N3_C4",
    "L_afpC3_N3_H1",
    "L_afpC3_N3_N2",
    "L_afpC3_O2_C3",
    "L_afpC3_O2_H1",
    "L_afpC4_C3_H1",
    "L_afpC4_C3_N3",
    "L_afpC4_C3_O1",
    "L_afpC4_C3_O2",
    "L_afpC4_C4_N3",
    "L_afpC4_N3_C4",
    "L_afpC4_N3_H1",
    "L_afpC4_O2_H1",
    "L_afpC4_P4_O1",
    "L_afpC4_P4_O2",
    "L_afpH1_C3_N2",
    "L_afpH1_C3_N3",
    "L_afpH1_C3_O1",
    "L_afpH1_C4_H1",
    "L_afpH1_C4_N3",
    "L_afpH1_C4_O2",
    "L_afpH1_C4_P4",
    "L_afpH1_N3_H1",
    "L_afpH1_N3_N2",
    "L_afpH1_O2_P4",
    "L_afpN2_C3_N2",
    "L_afpN2_N2_N3",
    "L_afpN2_N3_N2",
    "L_afpN3_C3_O1",
    "L_afpO1_C3_O2",
    "L_afpO1_P4_O2",
    "L_afpO2_P4_O2",
    "L_efpfam_acrylate",
    "L_efpfam_ketone",
    "L_efpfam_polyamides",
    "L_efpfam_single",
    "L_efpnorm_mol_wt",
    "L_efpnumatoms_none_H",
    "L_efpring",
    "L_mfpChi0n",
    "L_mfpChi0v",
    "L_mfpChi1n",
    "L_mfpChi1v",
    "L_mfpChi2n",
    "L_mfpChi2v",
    "L_mfpHallKierAlpha",
    "L_mfpMQNs13",
    "L_mfpMQNs14",
    "L_mfpMQNs15",
    "L_mfpMQNs16",
    "L_mfpMQNs17",
    "L_mfpMQNs19",
    "L_mfpMQNs20",
    "L_mfpMQNs21",
    "L_mfpMQNs26",
    "L_mfpMQNs27",
    "L_mfpMQNs28",
    "L_mfpMQNs29",
    "L_mfpMQNs30",
    "L_mfpMQNs31",
    "L_mfpMQNs32",
    "L_mfpMQNs35",
    "L_mfpMQNs36",
    "L_mfpMQNs40",
    "L_mfpMQNs41",
    "L_mfpMQNs42",
    "L_mfpNumAliphaticRings",
    "L_mfpNumAromaticRings",
    "L_mfptpsa",
    "L_linkermetalratio",
    "L_no",
    "L_noh",
    "L_nh2o",
]


def train_test_waterstability_baseline(train, test, num_trials: int = 100, seed: int = 42):
    X_train = train[features]
    X_test = test[features]

    y_train = train["stability_int"]
    y_test = test["stability_int"]

    baseline = XGBClassificationBaseline(num_trials=num_trials, seed=seed)
    baseline.tune(X_train, y_train)
    baseline.fit(X_train, y_train)

    xgb_predictions = baseline.predict(X_test)

    classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    classifier.fit(X_train, y_train)
    predictions, _ = classifier.predict(X_test, return_winning_probability=True)

    return {
        "xgboost": evaluate_classification(y_test, xgb_predictions),
        "tabpfn": evaluate_classification(y_test, predictions),
    }
