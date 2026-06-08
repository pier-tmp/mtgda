from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from inference import MtgDraftAssistant

asst = MtgDraftAssistant.from_pretrained()

# Stateless
pack_history = [
    ["Lightning Bolt", "Llanowar Elves", "Cancel"],    
    ["Shock", "Giant Growth", "Murder", "Consider"],   
]
pool = ["Lightning Bolt"]
for name, prob in asst.rank(pack_history, pool):
    print(f"  {prob:6.1%}  {name}")

# Stateful
d = asst.new_draft()
d.see(["Lightning Bolt", "Llanowar Elves", "Cancel"])
d.pick("Lightning Bolt")
print("pack 2 ranking:")
for name, prob in d.see(["Shock", "Giant Growth", "Murder", "Consider"]):
    print(f"  {prob:6.1%}  {name}")

# Custom cards
custom = {
    "name": "Made-Up Firebrand",
    "type_line": "Creature — Elemental",
    "mana_cost": "{1}{R}",
    "power": "3", "toughness": "1",
    "oracle_text": "Haste. When Made-Up Firebrand enters, it deals 2 damage to any target.",
}
pack = [custom, "Giant Growth", "Cancel"]
for name, prob in asst.rank([pack], pool=[]):
    print(f"  {prob:6.1%}  {name}")
