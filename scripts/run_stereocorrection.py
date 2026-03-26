"""
Python script for retrosynthesis tasks to run RetroTide with the post processing
stereochemistry correction KR Swaps. Predicts a chimeric type I PKS design and
PKS product that has maximal structure and stereochemistry correspondence with the
target molecule.

Requires configuration file in YAML format specifying:
- job_name: title for the retrosynthesis job
- output_dir: directory to save results
- molecule: user-defined target molecule in SMILES format
- starter_codes: loading module substrates to consider
- extender_codes: extender module substrates to consider
- stereo: user-specification of which stereochemistry to correct for (R/S, E/Z, all, none)
- offload_mech: TE mediated offloading mechanism (thiolysis, cyclization)

Returns:
- JSON file with imformation such as the final PKS design, PKS product SMILES, and
  similarity metrics for comparing the PKS product to the target molecule.
- SVG files providing a visual depiction of the stereochemistry correspondence between
  the PKS product and the target molecule, before and after the stereochemistry correction.
"""
from typing import Optional, List
import yaml
import bcs

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
config = load_config("../RetroTideV2/krswaps/config.yaml")

def modify_bcs_starters_extenders(starter_codes: Optional[List[str]] = None,
                                  extender_codes: Optional[List[str]] = None):
    """
    Modifies the starter and extender units available for RetroTide.
    Removes all starter and extender units not specifed in the input lists.
    """
    for key in list(bcs.starters.keys()):
        if key not in starter_codes:
            bcs.starters.pop(key, None)
    for key in list(bcs.extenders.keys()):
        if key not in extender_codes:
            bcs.extenders.pop(key, None)

modify_bcs_starters_extenders(starter_codes = config["starter_codes"],
                              extender_codes = config["extender_codes"])

from krswaps import krswaps as krs

results = krs.krswaps_stereo_correction(config['molecule'], config['stereo'], config['offload_mech'])
krs.output_results(results, config['job_name'], config['output_dir'])
