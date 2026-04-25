
# Initialize gmsh and create a new model:
import gmsh
gmsh.initialize()
gmsh.model.add("Rectangle_with_Disks")
dx_max = 0.14
dx_min = 0.07

# Set global options:
gmsh.option.setNumber("Geometry.OldNewReg", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", dx_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dx_max)
occ = gmsh.model.occ

# Construct boundary points:
xs, ys = [0.0, 7.0], [0.0, 7.0]
pts = {(ii, jj): occ.addPoint(xx, yy, 0.0, 0.0) for jj, yy in enumerate(ys) for ii, xx in enumerate(xs)}

# Define bottom, top, left and right lines:
l0 = occ.addLine(pts[(0, 0)], pts[(1, 0)])
l1 = occ.addLine(pts[(1, 0)], pts[(1, 1)])
l2 = occ.addLine(pts[(1, 1)], pts[(0, 1)])
l3 = occ.addLine(pts[(0, 1)], pts[(0, 0)])

# Define background:
loop = occ.addCurveLoop([l0, l1, l2, l3])
s_background = occ.addPlaneSurface([loop])

# Define obstacles:
x_all = [1.5, 5.5, 2.5, 4.5, 1.5, 5.5, 2.5, 4.5, 1.5, 3.5, 5.5]
y_all = [5.5, 5.5, 4.5, 4.5, 3.5, 3.5, 2.5, 2.5, 1.5, 1.5, 1.5]
s_all = [occ.addRectangle(x0 - 0.5, y0 - 0.5, 0.0, 1.0, 1.0)
         for x0, y0 in zip(x_all, y_all)]

# Define source:
s_source = occ.addRectangle(3.0, 3.0, 0.0, 1.0, 1.0)
occ.synchronize()

# Boolean fragments:
objectDimTags = [(2, s_background)]
toolDimTags   = [(2, s) for s in s_all] + [(2, s_source)]
outDimTags, outDimTagsMap = occ.fragment(objectDimTags, toolDimTags, removeObject=True, removeTool=True)
occ.synchronize()

# ---- Physical groups (labels) ----
def only_dim(dimtags, dim=2):
    return [t for d, t in dimtags if d == dim]

# Obstacles = union of fragments of each rectangle in s_all
obs_tags = set()
for i in range(len(s_all)): obs_tags.update(only_dim(outDimTagsMap[i+1], dim=2))
pg_obs = gmsh.model.addPhysicalGroup(2, sorted(obs_tags))
gmsh.model.setPhysicalName(2, pg_obs, "obstacles")

# Source = fragments of s_source (last tool)
src_tags = set(only_dim(outDimTagsMap[-1], dim=2))
pg_src = gmsh.model.addPhysicalGroup(2, sorted(src_tags))
gmsh.model.setPhysicalName(2, pg_src, "source")

# Background = everything except obstacles (but includes source automatically)
all_surf_tags = set(tag for (dim, tag) in gmsh.model.getEntities(2))
bg_tags = all_surf_tags - obs_tags - src_tags
pg_bg  = gmsh.model.addPhysicalGroup(2, sorted(bg_tags))
gmsh.model.setPhysicalName(2, pg_bg, "background")

# Generate the mesh:
gmsh.model.mesh.generate(2)
gmsh.write("square.msh")
gmsh.finalize()