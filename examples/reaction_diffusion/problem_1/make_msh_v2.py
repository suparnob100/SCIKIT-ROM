import gmsh

gmsh.initialize()
gmsh.model.add("Problem_1_mesh")

dx_max = 0.14/4
dx_min = 0.07/4

gmsh.option.setNumber("Geometry.OldNewReg", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", dx_min)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dx_max)

# Write legacy MSH2.2 for better downstream compatibility
# gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

occ = gmsh.model.occ

# ------------------------------------------------------------------
# Outer square
# ------------------------------------------------------------------
xs, ys = [0.0, 7.0], [0.0, 7.0]
pts = {(ii, jj): occ.addPoint(xx, yy, 0.0, 0.0)
       for jj, yy in enumerate(ys)
       for ii, xx in enumerate(xs)}

l0 = occ.addLine(pts[(0, 0)], pts[(1, 0)])
l1 = occ.addLine(pts[(1, 0)], pts[(1, 1)])
l2 = occ.addLine(pts[(1, 1)], pts[(0, 1)])
l3 = occ.addLine(pts[(0, 1)], pts[(0, 0)])

loop = occ.addCurveLoop([l0, l1, l2, l3])
s_background = occ.addPlaneSurface([loop])

# ------------------------------------------------------------------
# Obstacles
# ------------------------------------------------------------------
x_all = [1.5, 5.5, 2.5, 4.5, 1.5, 5.5, 2.5, 4.5, 1.5, 3.5, 5.5]
y_all = [5.5, 5.5, 4.5, 4.5, 3.5, 3.5, 2.5, 2.5, 1.5, 1.5, 1.5]

s_all = [occ.addRectangle(x0 - 0.5, y0 - 0.5, 0.0, 1.0, 1.0)
         for x0, y0 in zip(x_all, y_all)]

# ------------------------------------------------------------------
# Source
# ------------------------------------------------------------------
s_source = occ.addRectangle(3.0, 3.0, 0.0, 1.0, 1.0)

occ.synchronize()

# ------------------------------------------------------------------
# Boolean fragmentation
# ------------------------------------------------------------------
objectDimTags = [(2, s_background)]
toolDimTags = [(2, s) for s in s_all] + [(2, s_source)]

outDimTags, outDimTagsMap = occ.fragment(
    objectDimTags,
    toolDimTags,
    removeObject=True,
    removeTool=True
)
occ.synchronize()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def only_dim(dimtags, dim=2):
    return [t for d, t in dimtags if d == dim]

# ------------------------------------------------------------------
# 2D physical groups = materials
# ------------------------------------------------------------------
# material 1 = all obstacles
obs_tags = set()
for i in range(len(s_all)):
    obs_tags.update(only_dim(outDimTagsMap[i + 1], dim=2))
obs_tags = sorted(obs_tags)

pg_mat1 = gmsh.model.addPhysicalGroup(2, obs_tags)
gmsh.model.setPhysicalName(2, pg_mat1, "mat1")

# material 2 = source block
src_tags = sorted(set(only_dim(outDimTagsMap[-1], dim=2)))
pg_mat2 = gmsh.model.addPhysicalGroup(2, src_tags)
gmsh.model.setPhysicalName(2, pg_mat2, "mat2")

# material 3 = remaining background
all_surf_tags = set(tag for (dim, tag) in gmsh.model.getEntities(2))
bg_tags = sorted(all_surf_tags - set(obs_tags) - set(src_tags))

pg_mat3 = gmsh.model.addPhysicalGroup(2, bg_tags)
gmsh.model.setPhysicalName(2, pg_mat3, "mat3")

# ------------------------------------------------------------------
# 1D physical group = outer boundary
# ------------------------------------------------------------------
pg_bnd = gmsh.model.addPhysicalGroup(1, [l0, l1, l2, l3])
gmsh.model.setPhysicalName(1, pg_bnd, "boundary")

# ------------------------------------------------------------------
# Mesh and save
# ------------------------------------------------------------------
gmsh.model.mesh.generate(2)
gmsh.write("Problem_1.msh")
gmsh.finalize()