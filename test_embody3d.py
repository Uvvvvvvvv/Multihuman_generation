from tridi.data.embody3d_h2h_dataset import Embody3DH2HDataset

dataset = Embody3DH2HDataset(
    root="/media/uv/Data/workspace/tridi/embody-3d/datasets"
)

print("Total sequences:", len(dataset))

sample = dataset[0]
print("===== BatchData sample =====")

# Human1
print("Human1 shape:", sample.sbj_shape.shape)
print("Human1 pose:", sample.sbj_pose.shape)
print("Human1 global:", sample.sbj_global.shape)
print("Human1 transl:", sample.sbj_c.shape)

# Human2 (mirror human)
print("Human2 shape:", sample.obj_shape.shape)
print("Human2 pose:", sample.obj_pose.shape)
print("Human2 global:", sample.obj_global.shape)
print("Human2 transl:", sample.obj_c.shape)
