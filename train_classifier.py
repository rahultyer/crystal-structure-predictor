"""
Train a composition -> crystal structure classifier on Materials Project data.

Predicts two labels from composition alone:
  * crystal system (7 classes)
  * space group number (top-N most common groups, rest = "other")

Usage:
    export MP_API_KEY=your_key            # free key from https://next-gen.materialsproject.org/api
    python train_classifier.py            # writes models/structure_classifier.joblib
"""
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from matminer.featurizers.base import MultipleFeaturizer
from mp_api.client import MPRester
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TOP_SPACEGROUPS = 40          # keep the 40 most common space groups as classes
MAX_E_ABOVE_HULL = 0.05       # eV/atom: keep (near-)stable entries only


def download_data() -> pd.DataFrame:
    """Pull composition + symmetry for (near-)stable materials."""
    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, MAX_E_ABOVE_HULL),
            fields=["material_id", "formula_pretty", "symmetry", "energy_above_hull"],
        )
    rows = [{
        "material_id": str(d.material_id),
        "formula": d.formula_pretty,
        "crystal_system": str(d.symmetry.crystal_system),
        "spacegroup": int(d.symmetry.number),
        "e_hull": d.energy_above_hull,
    } for d in docs]
    df = pd.DataFrame(rows)
    # one row per formula: keep the lowest-energy polymorph (the ground state)
    df = df.sort_values("e_hull").drop_duplicates("formula", keep="first").reset_index(drop=True)
    print(f"{len(df)} unique compositions")
    return df


def build_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),   # statistics of elemental properties
        Stoichiometry(),                          # norms of the stoichiometry vector
        ValenceOrbital(props=["frac"]),           # s/p/d/f valence electron fractions
    ])


def featurize(df: pd.DataFrame, featurizer: MultipleFeaturizer) -> np.ndarray:
    comps = [Composition(f) for f in df["formula"]]
    X = featurizer.featurize_many(comps, ignore_errors=True, pbar=True)
    return np.nan_to_num(np.array(X, dtype=float))


def main():
    df = download_data()
    common = [sg for sg, _ in Counter(df["spacegroup"]).most_common(TOP_SPACEGROUPS)]
    df["sg_label"] = df["spacegroup"].where(df["spacegroup"].isin(common), other=0)  # 0 = "other"

    featurizer = build_featurizer()
    X = featurize(df, featurizer)

    models = {}
    for target in ["crystal_system", "sg_label"]:
        y = df[target].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=0)
        clf.fit(X_tr, y_tr)
        print(f"\n=== {target} ===")
        print(classification_report(y_te, clf.predict(X_te), zero_division=0))
        models[target] = clf

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "featurizer": featurizer,
        "crystal_system": models["crystal_system"],
        "spacegroup": models["sg_label"],
        "feature_labels": featurizer.feature_labels(),
    }, "models/structure_classifier.joblib")
    print("Saved models/structure_classifier.joblib")


if __name__ == "__main__":
    main()
