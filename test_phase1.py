import logging
import antigravity_data
from antigravity_data import DataFusionEngine, CrystalKnowledgeGraph, DefectGenerator

# Configure logging to show info
logging.basicConfig(level=logging.INFO)

def test_workflow():
    print("=== Testing DataFusionEngine ===")
    fusion = DataFusionEngine()
    targets = ["MoS2", "WS2"]
    
    # Test 1: Fetch Properties
    props = fusion.fetch_c2db_properties(targets)
    print(f"Fetched Properties: {props}")
    assert len(props) == 2
    
    # Test 2: Fetch Structures
    structs = fusion.fetch_materials_project_structure(targets)
    print(f"Fetched Structures for: {[s.formula for s in structs]}")
    assert len(structs) == 2
    
    print("\n=== Testing CrystalKnowledgeGraph ===")
    kg = CrystalKnowledgeGraph()
    kg.construct_from_fusion(fusion, targets)
    stats = kg.get_subgraph_stats()
    print(f"Graph Stats: {stats}")
    assert stats['nodes'] > 0
    assert stats['edges'] > 0
    
    print("\n=== Testing DefectGenerator ===")
    # Test 3: Vacancy
    vac_struct = DefectGenerator.generate_vacancy(structs[0], concentration=0.2)
    print(f"Original Sites: {len(structs[0].sites)}")
    print(f"Vacancy Sites: {len(vac_struct.sites)}")
    assert len(vac_struct.sites) < len(structs[0].sites)
    
    # Test 4: Substitution
    sub_struct = DefectGenerator.generate_substitution(structs[1], dopant="Se", concentration=0.5)
    dopant_present = any(s['species'] == 'Se' for s in sub_struct.sites)
    print(f"Dopant 'Se' present in {sub_struct.formula}: {dopant_present}")
    assert dopant_present
    
    print("\n=== Phase 1 Verification COMPLETE ===")

if __name__ == "__main__":
    test_workflow()
