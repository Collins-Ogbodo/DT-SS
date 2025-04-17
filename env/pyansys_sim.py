import json
import ansys.mechanical.core as pymech
import numpy as np
import os
import pandas as pd
from natsort import natsorted
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.preprocessing import normalize


"""   
    Mode Shape = { Mode 1 : 1st bending,
                    Mode 2 : 2nd bending,
                    Mode 3 : 1st torsional,
                    Mode 4 : 3rd bending,
                    Mode 5 : 1st Ridged body mode (z-direction),
                    Mode 6 : 2nd torsional,
                    Mode 7 : 4th bending,
                    Mode 8 : 5th bending,
                    Mode 9 : 3rd torsional,
                    Mode 10 : 4th torsional,
                    Mode 11 : 6th bending,
                    Mode 12 : 4th torsional,
                    Mode 13 : 7th bending,
                    Mode 14 : 1st bending (z-direction),
                    Mode 15 : 5th torsional,
                }
"""
class Cantilever():
    """  RL agent evironment for a Cantilever
         Dimension:
        Sensor : variable
        Material : Low alloy steel
        Fixed at one end
        Location : University of Sheffield 
        Reference : 
        section : 
    """
    def __init__(self, config: dict) -> None:
        super().__init__()
        #PyMechanical Environment Parameters
        self.geo_path = os.path.join(os.getcwd(),"env","Geometry","Cantilever-EMA.agdb")
        self.mat_path = os.path.join(os.getcwd(),"env","Material","ANSYS GRANTA-Low-Alloy-Steel-4140-Normalised.xml")
        self.mechanical = pymech.launch_mechanical(batch = True if config.get("render", False) == False else False, 
                                                   cleanup_on_exit= True)
        project_directory = self.mechanical.project_directory
        print(f'Project Directory = {project_directory}') #obtain working project directory of ansys mechnaical instance
        
        #Mode to add in mode shaoe matric
        self.sim_modes = np.array(config.get('sim_modes', [0,1,2])) 
        self.num_sensors = config.get("num_sensors", 4)

        #list of mode shape across environment
        self.phi_list = []
        self.correlation_covariance_matrix_list = []
        self.mode_shape_folder_name =  os.path.join(os.getcwd(), "env","Mode_Shape")
        
        #Inittialise
        self.mass = 0
        self.point_mass_node = 0 
        self.cond_count = 0
            
            
        def run() -> np.array:
            """ Run modal analysis"""
            natural_freq_script = """
#Run Modal Analysis
modal_analysis.Solution.Solve(True)
#modal_solution_modal.GetResults() 

#List of all model shape names for tracking
Directional_deformation = [Dir_def_1, Dir_def_2, Dir_def_3, Dir_def_4, Dir_def_5, 
                           Dir_def_6, Dir_def_7, Dir_def_8, Dir_def_9, Dir_def_10, Dir_def_11,
                           Dir_def_12, Dir_def_13, Dir_def_14, Dir_def_15]

#Absolute Directory
fileExtension = r".txt"
Natural_Frequency = {{}}
file_names = []

#Export Direction Deformation .txt file
for Dir_deform in Directional_deformation:
    file_names.append(Dir_deform.Name)
    path = os.path.join(cwd, "{folder:s}", "{point_mass_node:s}" + "-" + "{mass:s}"+ "-" + str(Dir_deform.Name) + fileExtension)
    Dir_deform.ExportToTextFile(path)
    Natural_Frequency[str(Dir_deform.Name)] = Dir_deform.ReportedFrequency.Value

json.dumps(Natural_Frequency)
        """
            natural_freq = self.mechanical.run_python_script(natural_freq_script.format(folder = self.mode_shape_folder_name,
                                                                                        point_mass_node = str(self.point_mass_node),
                                                                                        mass = str(self.mass)))
            wn = np.sort(list(json.loads(natural_freq).values()))
            wn = wn[self.sim_modes]
            print("Natural Frequencies", wn)
            return wn
        
        def make() -> None: 
            """Setup the Modal Analysis environment and product natural frequencies"""
            modal_analysis = """ 
import os
import json
cwd = os.getcwd()
#Import function
geometry_import_group = Model.GeometryImportGroup
geometry_import = geometry_import_group.AddGeometryImport()

#Import Geometry
geometry_import_format = Ansys.Mechanical.DataModel.Enums.GeometryImportPreference.Format.Automatic
geometry_import_preferences = Ansys.ACT.Mechanical.Utilities.GeometryImportPreferences()
geometry_import.Import(r"{geo_path:s}", geometry_import_format, geometry_import_preferences)

#Load engineering data source and 
materials = Model.Materials
imported_materials = materials.Import(r"{mat_path:s}")

#Unit
ExtAPI.Application.ActiveUnitSystem = MechanicalUnitSystem.StandardMKS

#Set Geometry Material
material_assignment = Model.Materials.AddMaterialAssignment()
material_assignment.Material = imported_materials[0].Name
sel = ExtAPI.SelectionManager.CreateSelectionInfo(Ansys.ACT.Interfaces.Common.SelectionTypeEnum.GeometryEntities)
sel.Ids = [body.GetGeoBody().Id for body in Model.Geometry.GetChildren(Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory.Body, True)]
material_assignment.Location = sel

#Set mesh Element size
mesh = Model.Mesh
mesh.ElementSize = Quantity(0.005, "m")
#Generate Mesh
mesh.GenerateMesh()
#View Mesh Quality
mesh.MeshMetric = MeshMetricType.ElementQuality

#Add Modal Analysis
#region Toolbar Action
model = Model
modal_analysis = model.AddModalAnalysis()
modal_analysis_settings = DataModel.GetObjectsByType(Ansys.ACT.Automation.Mechanical.AnalysisSettings.ANSYSAnalysisSettings)

#Set Number of Modes
modal_analysis_settings = DataModel.GetObjectById(modal_analysis_settings[0].ObjectId)
num_mode = 15
modal_analysis_settings.MaximumModesToFind = num_mode

#damping
modal_analysis_settings.Damped = True
modal_analysis_settings.StructuralDampingCoefficient = 0.015

#Add Fixed End
fixed_support = modal_analysis.AddFixedSupport()

#Name Selection 
support_face = ExtAPI.DataModel.GetObjectsByName("FixedSupport")[0].Ids
#Select fixed end surface
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
selection.Ids = support_face
fixed_support.Location = selection
 
#Set solver config
config = ExtAPI.Application.SolveConfigurations["My Computer"]
config.SolveProcessSettings.MaxNumberOfCores = 4
config.SolveProcessSettings.DistributeSolution = True

#Run Modal Analysis
modal_analysis.Solution.Solve(True)
modal_analysis_solution = DataModel.GetObjectsByType(Ansys.ACT.Automation.Mechanical.Solution)


#Mode Shape Result/deformation
modal_solution_modal =  modal_analysis.Solution

#Add Directional Deformation (Mode Shapes)
Dir_def_1  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_1.NormalOrientation = NormalOrientationType.YAxis
Dir_def_1.Mode =1 

Dir_def_2  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_2.NormalOrientation = NormalOrientationType.YAxis
Dir_def_2.Mode =2 

Dir_def_3  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_3.NormalOrientation = NormalOrientationType.YAxis
Dir_def_3.Mode =3 

Dir_def_4  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_4.NormalOrientation = NormalOrientationType.YAxis
Dir_def_4.Mode =4 

Dir_def_5  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_5.NormalOrientation = NormalOrientationType.YAxis
Dir_def_5.Mode =5 

Dir_def_6  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_6.NormalOrientation = NormalOrientationType.YAxis
Dir_def_6.Mode =6 

Dir_def_7  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_7.NormalOrientation = NormalOrientationType.YAxis
Dir_def_7.Mode =7 

Dir_def_8  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_8.NormalOrientation = NormalOrientationType.YAxis
Dir_def_8.Mode =8 

Dir_def_9  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_9.NormalOrientation = NormalOrientationType.YAxis
Dir_def_9.Mode =9 

Dir_def_10  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_10.NormalOrientation = NormalOrientationType.YAxis
Dir_def_10.Mode =10 

Dir_def_11  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_11.NormalOrientation = NormalOrientationType.YAxis
Dir_def_11.Mode =11 

Dir_def_12  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_12.NormalOrientation = NormalOrientationType.YAxis
Dir_def_12.Mode =12 

Dir_def_13  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_13.NormalOrientation = NormalOrientationType.YAxis
Dir_def_13.Mode =13

Dir_def_14  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_14.NormalOrientation = NormalOrientationType.YAxis
Dir_def_14.Mode =14 

Dir_def_15  = modal_solution_modal.AddDirectionalDeformation()
Dir_def_15.NormalOrientation = NormalOrientationType.YAxis
Dir_def_15.Mode =15 

         """

            _ = self.mechanical.run_python_script(modal_analysis.format(geo_path = self.geo_path,
                                                                                    mat_path = self.mat_path)) 
            self.wn = run()
            return None
        
        def localisation_damage_cases(point_mass_node : list, conds_mass: list):
            #Add mass point 
            add_damage_script = """ 
#Add mass point 
if {count} == 0:
    geometry = Model.Geometry
    #Pont mass
    point_mass = geometry.AddPointMass()
    point_mass.AppliedBy = RemoteApplicationType.DirectAttachment
selection_point = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
selection_point.Ids = [{point_mass_node}]
point_mass.Location = selection_point
point_mass.Mass = Quantity({mass}, "kg")
            """
            for point_mass_node, mass in zip(point_mass_node, conds_mass):
                _ = self.mechanical.run_python_script(add_damage_script.format(point_mass_node = point_mass_node,
                                                                            mass = mass,
                                                                            count = self.cond_count)) 
                self.point_mass_node = point_mass_node
                self.mass = mass
                self.run()
                self.phi = extract_mode_shape(self.observation_space_node).T
                self.phi = normalize(self.phi, axis=0, norm='max') if config.get("norm", False) else self.phi
                self.cond_count+=1
                self.phi_list.append(self.phi)
                cor_cov_ma = correlation_covariance_matrix(self.phi)
                self.correlation_covariance_matrix_list.append(cor_cov_ma)
            return None
        
        def severity_damage_cases(point_mass_node : list, conds_mass: list):
            #Add mass point 
            add_damage_script = """ 
#Add mass point 
if {count} == 0:
    geometry = Model.Geometry
#Pont mass
point_mass = geometry.AddPointMass()
point_mass.AppliedBy = RemoteApplicationType.DirectAttachment
selection_point = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
selection_point.Ids = [{point_mass_node}]
point_mass.Location = selection_point
point_mass.Mass = Quantity({mass}, "kg")
            """
            for point_mass_node, mass in zip(point_mass_node, conds_mass):
                _ = self.mechanical.run_python_script(add_damage_script.format(point_mass_node = point_mass_node,
                                                                            mass = mass,
                                                                            count = self.cond_count)) 
                self.point_mass_node = point_mass_node
                self.mass = mass
                self.run()
                self.phi = extract_mode_shape(self.observation_space_node).T
                self.phi = normalize(self.phi, axis=0, norm='max') if config.get("norm", False) else self.phi
                self.cond_count+=1
                self.phi_list.append(self.phi)
                cor_cov_ma = correlation_covariance_matrix(self.phi)
                self.correlation_covariance_matrix_list.append(cor_cov_ma)
            return None
        
        def observation_space_node() -> np.array: 
            """Get all avaliable nodes Ids where sensor and shaker can be placed"""
            observation = self.mechanical.run_python_script(""" 
#Name Slection for search suraface
def nodeIds(MeshData, NamedSelection):
    faceId = ExtAPI.DataModel.GetObjectsByName(NamedSelection)[0].Ids[0]
    return MeshData.MeshRegionById(faceId).NodeIds

#Mesh Data
mesh_data = ExtAPI.DataModel.Project.Model.Analyses[0].MeshData                                             

#Get nodes space based on name selection                                                              
nodes = {'Surface_nodes' : str(nodeIds(mesh_data, "Surface_nodes"))    
        }
json.dumps(nodes)
            """)
            data = json.loads(observation)
            node_space = np.array(eval(data['Surface_nodes'].replace("List[int]", ""))[:1462])
            return node_space
            
        def coordinates()-> np.array:
            #Extract all node coordinates [Id, X, Y, Z]
            nodes_coord = self.mechanical.run_python_script("""                                                                                                                  
#Mesh Data
mesh_data = ExtAPI.DataModel.Project.Model.Analyses[0].MeshData                                             
all_node_coord = []
def nodeIds(MeshData, NamedSelection):
    faceId = ExtAPI.DataModel.GetObjectsByName(NamedSelection)[0].Ids[0]
    return MeshData.MeshRegionById(faceId).NodeIds
named_selection = ["Surface_nodes"]                                                    
for name in named_selection :  #loop through nodes
    all_node_coord.append([[Id, mesh_data.NodeById(Id).X, mesh_data.NodeById(Id).Y, mesh_data.NodeById(Id).Z] for Id in nodeIds(mesh_data, name)])
                                                                                                                                              
json.dumps({'all_node_coord' : all_node_coord} )
                """)
            return np.array(list(json.loads(nodes_coord).values()))
        
        def edge_node()-> tuple:
            #Extract edge node
            edge_node_script = """ 
def edge_nodeIds(MeshData, NamedSelection):
    placement_face_Act = ExtAPI.DataModel.GetObjectsByName(NamedSelection)[0]
    face_edges = placement_face_Act.Entities[0].Edges
    return MeshData.MeshRegionById(face_edges[{index}].Id).NodeIds
                                                                                                               
#Mesh Data
mesh_data = ExtAPI.DataModel.Project.Model.Analyses[0].MeshData                                             
#Get nodes space based on name selection                                                              
nodes = edge_nodeIds(mesh_data, "Surface_nodes")
json.dumps({{'nodes': str(nodes)}})
                """
            bottom_edge_nodes_json = self.mechanical.run_python_script(""" 
def edge_nodeIds(MeshData, NamedSelection):
    placement_face_Act = ExtAPI.DataModel.GetObjectsByName(NamedSelection)[0]
    face_edges = placement_face_Act.Entities[0].Edges
    return MeshData.MeshRegionById(face_edges[0].Id).NodeIds
                                                                                                               
#Mesh Data
mesh_data = ExtAPI.DataModel.Project.Model.Analyses[0].MeshData                                             
#Get nodes space based on name selection                                                              
nodes = edge_nodeIds(mesh_data, "Surface_nodes")
json.dumps({'nodes': str(nodes)})
                    """)
            left_edge_nodes = json.loads(self.mechanical.run_python_script(edge_node_script.format(index = 2)))
            right_edge_nodes = json.loads(self.mechanical.run_python_script(edge_node_script.format(index = 3)))
            top_edge_nodes = json.loads(self.mechanical.run_python_script(edge_node_script.format(index = 1)))
            bottom_edge_nodes= json.loads(bottom_edge_nodes_json)
            left_edge_nodes = np.array(eval(left_edge_nodes['nodes'].replace("List[int]", "")))
            right_edge_nodes = np.array(eval(right_edge_nodes['nodes'].replace("List[int]", ""))) 
            top_edge_nodes = np.array(eval(top_edge_nodes['nodes'].replace("List[int]", "")))
            bottom_edge_nodes = np.array(eval(bottom_edge_nodes['nodes'].replace("List[int]", "")))
            return (left_edge_nodes, right_edge_nodes, top_edge_nodes, bottom_edge_nodes)

        
        def render(current_state : np.array) -> None:
            """Render method 
            -----------------
            Arg: mode: Render type
            ----------------
            return: None"""
            render_script = """
my_selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
my_selection.Ids = {node_array}
ExtAPI.SelectionManager.ClearSelection()
ExtAPI.SelectionManager.NewSelection(my_selection)
            """
            node_list = list(current_state)
            self.mechanical.run_python_script(render_script.format(node_array = node_list))
            return None
        
        def extract_mode_shape(node: np.array) -> np.array:
            """Extract node mode shape from text file """
            # Get all file names in the folder
            file_names = [name for name in os.listdir(self.mode_shape_folder_name) if name.endswith(".txt") and name.startswith(str(self.point_mass_node))]
            file_names = natsorted(file_names, key=lambda x: (x != str(self.point_mass_node) + "-" + str(self.mass)+ "-" + 'Directional Deformation.txt', x))
            
            # Read and combine mode shapes from files
            mode_shape_sim = pd.concat([pd.read_csv(os.path.join(self.mode_shape_folder_name, name), sep='\t', usecols=[1]) for name in file_names], axis=1)
            mode_shape_sim = mode_shape_sim.loc[node-1].to_numpy().T #-1 because of python indexing
            return mode_shape_sim[self.sim_modes, :]
        
        def close() -> None:
            self.mechanical.clear()
            self.mechanical.exit(force= True)
            return None   
        
        #Initialise the evironment setup
        make()
        self.render = render        
        self.close = close
        self.run = run
        self.left_edge_nodes, self.right_edge_nodes, self.top_edge_nodes, self.bottom_edge_nodes = edge_node()
        self.observation_space_node = observation_space_node()
        self.nodes_coord = coordinates()
        self.phi = extract_mode_shape(self.observation_space_node).T
        self.phi = normalize(self.phi, axis=0, norm='max') if config.get("norm", False) else self.phi
        self.phi_list.append(self.phi)
        self.extract_mode_shape = extract_mode_shape
        #coordinate Initialisation
        self.coord_2d_array = np.round(
            self.nodes_coord.reshape(
                self.nodes_coord.shape[-2], 
                self.nodes_coord.shape[-1]),4) #Reshape from 4D to 2D row-wise
            
        """
        Computes the Euclidean distance between sensors for penalty calculations.
        """
        active_node_indice = [np.where(self.coord_2d_array[:,0] == ids)[0][0] for ids in self.observation_space_node] #Exclude mid point node id
        active_node_coord = self.coord_2d_array[active_node_indice, 1:]
        self.norm2_dist = squareform(pdist(active_node_coord, 'euclidean'))
        
        
        def correlation_covariance_matrix(phi) -> np.array:
            """
            Computes the correlation-covariance matrix combining modal correlation and spatial proximity.
            Reference:
                [1] Kim, Seon-Hu, and Chunhee Cho. "Effective independence in optimal sensor placement associated with general Fisher information 
                    involving full error covariance matrix." Mechanical Systems and Signal Processing 212 (2024): 111263.
                [2] Vincenzi, Loris, and Laura Simonini. "Influence of model errors in optimal sensor placement." Journal of Sound and Vibration 
                    389 (2017): 119-133.
            """
            
            num_nodes = len(self.observation_space_node)
            num_modes = phi.shape[1]
        
            # Precompute max values for each pair of nodes across all modes
            abs_phi = np.abs(phi)
            # Reshape to enable broadcasting: (num_nodes, num_modes, 1) and (1, num_modes, num_nodes)
            max_vals = np.maximum(abs_phi[:, :, np.newaxis], abs_phi.T[np.newaxis, :, :])
            
            # Initialize the mode shape correlation matrix
            mode_shape_corr = np.ones((num_nodes, num_nodes))
            
            # Iterate over all unique pairs of nodes
            for i, j in combinations(range(num_nodes), 2):
                # Compute the ratios for node i and j
                ratio_i = np.divide(abs_phi[i], max_vals[i, :, j], out=np.ones_like(abs_phi[i]), where=max_vals[i, :, j] != 0)
                ratio_j = np.divide(abs_phi[j], max_vals[j, :, i], out=np.ones_like(abs_phi[j]), where=max_vals[j, :, i] != 0)
                
                # Compute the correlation factor and fill both symmetric entries
                correlation = np.dot(ratio_i, ratio_j) / num_modes
                mode_shape_corr[i, j] = correlation
                mode_shape_corr[j, i] = correlation
            
            # Compute spatial correlation distance factor
            sp_cor_len = 0.42981 / self.num_sensors
            distance_factor = np.exp(-self.norm2_dist / sp_cor_len)
            
            # Element-wise multiplication of mode shape correlation and distance factor
            correlation_covariance_matrix = mode_shape_corr * distance_factor
            
            return correlation_covariance_matrix
        self.correlation_covariance_matrix = correlation_covariance_matrix(self.phi)
        
        self.correlation_covariance_matrix_list.append(self.correlation_covariance_matrix)
        #Extract list of modeshape and covariance matrices
        if config.get("condition_case") == 'localisation': 
            localisation_damage_cases(config.get("node_id"),
                                      config.get("mass"))
        elif config.get("condition_case") == 'severity': 
            severity_damage_cases(config.get("node_id"),
                                  config.get("mass"))
        
        

        


    
        
        