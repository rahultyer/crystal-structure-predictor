"""
Crystal structure prediction API.

  POST /predict/class      composition -> likely crystal system + space groups (trained classifier)
  POST /predict/structure  composition (+ pressure) -> ranked, relaxed candidate structures with energies

Pipeline for /predict/structure:
  1. Candidate generation: find known structures with the same anonymous formula (e.g. ABC3)
     in Materials Project and substitute the user's elements (prototype substitution).
  2. Relaxation: CHGNet machine-learned interatomic potential, cell + positions, at the given pressure.
  3. Ranking: enthalpy H = E + PV per atom; at 0 GPa also energy above the MP convex hull.

Run:
    pip install -r requirements.txt
    export MP_API_KEY=your_key
    uvicorn app:app --reload
"""
import os
from functools import lru_cache

import joblib
import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.units import GPa
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mp_api.client import MPRester
from pydantic import BaseModel, Field
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

app = FastAPI(title="Crystal structure predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "models/structure_classifier.joblib"


# ---------- shared resources (loaded once) ----------
@lru_cache
def classifier():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(503, "Classifier not trained yet. Run train_classifier.py first.")
    return joblib.load(MODEL_PATH)


@lru_cache
def chgnet():
    return CHGNet.load()


def mp():
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise HTTPException(503, "Set the MP_API_KEY environment variable.")
    return MPRester(key)


def parse(formula: str) -> Composition:
    try:
        comp = Composition(formula)
    except Exception:
        raise HTTPException(422, f"Could not read '{formula}' as a chemical formula.")
    if len(comp) == 0:
        raise HTTPException(422, "Empty formula.")
    return comp.reduced_composition


# ---------- request / response models ----------
class ClassRequest(BaseModel):
    formula: str = Field(..., examples=["SrTiO3"])


class StructureRequest(BaseModel):
    formula: str = Field(..., examples=["SrTiO3"])
    pressure_gpa: float = Field(0.0, ge=0, le=300)
    max_candidates: int = Field(8, ge=1, le=30)
    fmax: float = Field(0.05, gt=0, description="Force convergence, eV/Å")
    steps: int = Field(400, ge=10, le=2000)


# ---------- 1. classifier ----------
@app.post("/predict/class")
def predict_class(req: ClassRequest):
    comp = parse(req.formula)
    bundle = classifier()
    x = np.nan_to_num(np.array([bundle["featurizer"].featurize(comp)], dtype=float))

    def top(model, k=3):
        p = model.predict_proba(x)[0]
        idx = np.argsort(p)[::-1][:k]
        return [{"label": str(model.classes_[i]), "probability": round(float(p[i]), 3)} for i in idx]

    sg = top(bundle["spacegroup"], 5)
    for s in sg:
        s["label"] = "other" if s["label"] == "0" else f"#{s['label']}"
    return {"formula": comp.reduced_formula, "crystal_system": top(bundle["crystal_system"]), "spacegroup": sg}


# ---------- 2. candidate generation by prototype substitution ----------
def element_map(proto_comp: Composition, target: Composition):
    """Map prototype elements onto target elements with equal amounts, ordered by electronegativity."""
    def key(c):
        return sorted(c.items(), key=lambda kv: (kv[1], kv[0].X))
    p, t = key(proto_comp.reduced_composition), key(target)
    if [round(a, 6) for _, a in p] != [round(a, 6) for _, a in t]:
        return None
    return {pe.symbol: te.symbol for (pe, _), (te, _) in zip(p, t)}


def candidates(comp: Composition, limit: int) -> list[dict]:
    anon = comp.anonymized_formula
    with mp() as m:
        docs = m.materials.summary.search(
            formula=anon,
            energy_above_hull=(0, 0.1),
            fields=["material_id", "formula_pretty", "structure", "symmetry", "energy_above_hull"],
        )
    # one prototype per space group, most stable first, to keep the search diverse
    docs = sorted(docs, key=lambda d: d.energy_above_hull)
    seen, out = set(), []
    for d in docs:
        sg = d.symmetry.number
        if sg in seen:
            continue
        mapping = element_map(d.structure.composition, comp)
        if mapping is None:
            continue
        s = d.structure.copy()
        s.replace_species(mapping)
        # rescale volume from the prototype's atoms to the new atoms (atomic-radius cubes)
        old = sum((Element(e).atomic_radius or 1.5) ** 3 * n for e, n in d.structure.composition.get_el_amt_dict().items())
        new = sum((Element(e).atomic_radius or 1.5) ** 3 * n for e, n in s.composition.get_el_amt_dict().items())
        s.scale_lattice(s.volume * new / old)
        out.append({"structure": s, "prototype": f"{d.formula_pretty} ({d.material_id})", "proto_sg": sg})
        seen.add(sg)
        if len(out) >= limit:
            break
    return out


# ---------- 3. relaxation with CHGNet ----------
def relax(s: Structure, pressure_gpa: float, fmax: float, steps: int):
    atoms = AseAtomsAdaptor.get_atoms(s)
    atoms.calc = CHGNetCalculator(model=chgnet())
    cell_filter = FrechetCellFilter(atoms, scalar_pressure=pressure_gpa * GPa)
    FIRE(cell_filter, logfile=None).run(fmax=fmax, steps=steps)
    n = len(atoms)
    e_total = atoms.get_potential_energy()
    h_total = e_total + pressure_gpa * GPa * atoms.get_volume()
    return AseAtomsAdaptor.get_structure(atoms), e_total, e_total / n, h_total / n


# ---------- 4. ranking ----------
def hull_distance(comp: Composition, energy_total: float):
    """Energy above the MP convex hull (0 GPa).
    Note: CHGNet is trained on MP data, but for publication-quality numbers apply
    MaterialsProject2020Compatibility so corrections are consistent on both sides."""
    with mp() as m:
        entries = m.get_entries_in_chemsys([e.symbol for e in comp.elements])
    pd_ = PhaseDiagram(entries)
    _, e_hull = pd_.get_decomp_and_e_above_hull(PDEntry(comp, energy_total), allow_negative=True)
    return float(e_hull)


@app.post("/predict/structure")
def predict_structure(req: StructureRequest):
    comp = parse(req.formula)
    cands = candidates(comp, req.max_candidates)
    if not cands:
        raise HTTPException(404, f"No prototypes with the formula type {comp.anonymized_formula} found. "
                                 "Use a generative model (e.g. DiffCSP, MatterGen) or random structure search here.")
    results = []
    for c in cands:
        try:
            s, e_tot, e_atom, h_atom = relax(c["structure"], req.pressure_gpa, req.fmax, req.steps)
        except Exception as exc:  # a bad starting guess shouldn't kill the request
            results.append({"prototype": c["prototype"], "error": str(exc)})
            continue
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        refined = sga.get_conventional_standard_structure()
        row = {
            "prototype": c["prototype"],
            "spacegroup": sga.get_space_group_symbol(),
            "spacegroup_number": sga.get_space_group_number(),
            "crystal_system": sga.get_crystal_system(),
            "lattice": {k: round(v, 4) for k, v in zip(["a", "b", "c", "alpha", "beta", "gamma"], refined.lattice.parameters)},
            "density_g_cm3": round(float(s.density), 3),
            "energy_eV_per_atom": round(e_atom, 4),
            "enthalpy_eV_per_atom": round(h_atom, 4),
            "cif": str(CifWriter(refined, symprec=0.1)),
        }
        if req.pressure_gpa == 0:
            try:
                comp_cell = s.composition
                row["e_above_hull_eV_per_atom"] = round(hull_distance(comp_cell, e_tot), 4)
            except Exception:
                row["e_above_hull_eV_per_atom"] = None
        results.append(row)

    ok = sorted([r for r in results if "error" not in r], key=lambda r: r["enthalpy_eV_per_atom"])
    if ok:
        h0 = ok[0]["enthalpy_eV_per_atom"]
        for r in ok:
            r["relative_enthalpy_meV_per_atom"] = round(1000 * (r["enthalpy_eV_per_atom"] - h0), 1)
    return {
        "formula": comp.reduced_formula,
        "pressure_gpa": req.pressure_gpa,
        "ranked": ok,
        "failed": [r for r in results if "error" in r],
        "note": "Lowest enthalpy = predicted stable structure. Differences under ~25 meV/atom are within model error; verify with DFT.",
    }


@app.get("/health")
def health():
    return {"ok": True, "classifier_trained": os.path.exists(MODEL_PATH)}
