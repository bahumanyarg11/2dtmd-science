import sys
import os
import pandas as pd
import torch
import pyvista as pv

# Fix path
sys.path.append(os.getcwd())

print("--- 1. Testing Antigravity Data ---")
try:
    from antigravity_data import PublicDataEngine, RealLiteratureEngine #, VirtualScreeningEngine
    
    # Test Real C2DB
    pde = PublicDataEngine()
    df_c2db = pde.fetch_c2db_mirror()
    print(f"C2DB Real Data Loaded: {len(df_c2db)} rows")
    if len(df_c2db) == 0:
        print("ERROR: C2DB Data is empty")
        
    # Test Real Lit
    rle = RealLiteratureEngine()
    df_lit = rle.load_literature_data()
    print(f"Literature Data Loaded: {len(df_lit)} rows")
    if len(df_lit) == 0:
        print("ERROR: Literature Data is empty")
        
except Exception as e:
    print(f"FAIL: Antigravity Data Crash: {e}")
    import traceback
    traceback.print_exc()

print("\n--- 2. Testing Simulation Engine (PyVista) ---")
try:
    from frontend.simulation_engine import AntigravityEngine
    engine = AntigravityEngine()
    print("Engine Initialized")
    
    # Test Synthesis Twin (PyVista Grid)
    print("Running Synthesis Twin...")
    grid = engine.run_synthesis_twin(750, 10, 20)
    print(f"Grid Generated: {type(grid)}")
    
    if isinstance(grid, pv.ImageData):
        print("Success: Grid is pv.ImageData")
    else:
        print(f"Warning: Grid is {type(grid)}, expected ImageData/UniformGrid")

except Exception as e:
    print(f"FAIL: Simulation Engine Crash: {e}")
    import traceback
    traceback.print_exc()
