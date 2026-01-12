import torch
import numpy as np
import pyvista as pv
import os
from dtmd.core_ai.fno_synthesis import SynthesisFNO
from dtmd.core_ai.gflownet_agent import CrystalGFlowNet

class DTMDEngine:
    """
    The Physics Engine that drives the UI.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"--> Loading Engine on {self.device}...")
        
        # Load Phase 2 Models (In production, load weights here)
        self.fno = SynthesisFNO().to(device)
        self.agent = CrystalGFlowNet().to(device)
        
        # --- Checkpoint Loader ---
        ckpt_path = "dtmd_brain.pt"
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device)
                self.fno.load_state_dict(state)
                print(f"--> [BRAIN ACTIVE] Loaded trained weights from {ckpt_path}")
            except Exception as e:
                print(f"--> [WARNING] Failed to load brain: {e}")
        else:
            print("--> [SIMULATION MODE] No brain found. Using random initialization.")
            
        self.fno.eval()
        self.agent.eval()

    def run_synthesis_twin(self, temp, pressure, flow_rate):
        """
        Runs the FNO Digital Twin.
        Input: Scalar Physics parameters.
        Output: 3D Visualization Grid (PyVista object).
        """
        # 1. Prepare Input Tensor (Batch, GridX, GridY, Features)
        # Simulating a 64x64 sensor grid input
        grid_size = 64
        x = torch.zeros(1, grid_size, grid_size, 4).to(self.device)
        
        # Normalize inputs and fill grid
        x[:, :, :, 0] = temp / 1000.0
        x[:, :, :, 1] = pressure / 100.0
        x[:, :, :, 2] = flow_rate
        
        # KEY FIX: Channel 3 = Radial Distance (Geometry)
        # Allows model to understand "Center" vs "Edge"
        y, xx_grid = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size), indexing='ij')
        r_map = torch.sqrt(xx_grid**2 + y**2).to(self.device)
        x[:, :, :, 3] = r_map

        # 2. FNO Inference (The "Instant" CFD)
        with torch.no_grad():
            concentration_field = self.fno(x) # Output: (1, 64, 64, 1)

        # 3. Convert to PyVista Grid for Rendering
        data = concentration_field.squeeze().cpu().numpy()
        
        # Create a structured grid
        grid = pv.ImageData()
        grid.dimensions = (grid_size, grid_size, 1)
        grid.point_data["Gas Concentration"] = data.flatten(order="F")
        
        return grid

    def get_crystal_structure(self, material_name="MoS2"):
        """
        Generates 3D atomic structure.
        """
        # Placeholder for GFlowNet generation logic
        # In a real run, self.agent.sample() would give these coords
        
        # Creating a visually pleasing MoS2 lattice
        lattice = pv.PolyData()
        points = []
        colors = []
        
        # Generate Hexagonal Lattice
        rows, cols = 5, 5
        for i in range(rows):
            for j in range(cols):
                # Mo Atom (Layer 0)
                x = i * 3.16 + (1.58 if j % 2 else 0)
                y = j * 2.73
                z = 0
                points.append([x, y, z])
                colors.append(0) # 0 = Molybdenum

                # S Atoms (Layer +/- 1)
                points.append([x, y + 1.82, 1.5])
                colors.append(1) # 1 = Sulfur
                points.append([x, y + 1.82, -1.5])
                colors.append(1)

        lattice.points = np.array(points)
        lattice.point_data["AtomType"] = np.array(colors)
        return lattice
