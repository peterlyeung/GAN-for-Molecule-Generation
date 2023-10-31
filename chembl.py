import pandas as pd
from chembl_webresource_client.new_client import new_client

# Set up your authentication (you may need to obtain an API key)
config = {
    'username': 'your_username',
    'password': 'your_password'
}
# Create a new ChEMBL client
# client = new_client()

# Specify the target of interest (e.g., a protein target)
# target_name = 'Kinase PIM1'

target = new_client.target
activity = new_client.activity
herg = target.filter(pref_name__iexact='hERG').only('target_chembl_id')[0]
herg_activities = activity.filter(target_chembl_id=herg['target_chembl_id']).filter(standard_type="IC50")

# len(herg_activities)
df = pd.DataFrame.from_dict(herg_activities)


from rdkit import Chem
from rdkit.Chem import AllChem

# Generate a simple molecule using SMILES notation
new_molecule = Chem.MolFromSmiles('CCO')  # Example: Ethanol

# Compute descriptors for the new molecule
mol_descriptor = AllChem.Compute2DCoords(new_molecule)

# Visualize the molecule
img = Chem.Draw.MolToImage(new_molecule)
img.show()