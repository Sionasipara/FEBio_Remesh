
"""Create a uniform isotropic remesh of the FEBio breast model.

The utility extracts the exterior of the source ``tet4`` volume, uses Gmsh to
generate near-equilateral surface triangles and an optimized tetrahedral volume,
and writes the result back into a copy of the FEBio XML model.  Adipose,
fibroglandular, and tumour labels are transferred with an exact source-cell
lookup, with a nearest-centroid fallback for points affected by numerical surface
projection.  The ChestWall and posterior anchor node sets are reconstructed
geometrically, and a conforming membrane-equivalent ``tri3`` skin shell is rebuilt
on all exterior faces except the chest wall.

The original materials, analysis controls, boundary-condition definitions, loads,
and output requests are retained.  Existing FEBio result files are not reusable
after remeshing because node and element identities change.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REGIONS = ("adipose_tissue", "fibroglandular_tissue", "tumour")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a uniform Gmsh surface and tet4 volume mesh for the FEBio "
            "breast model while reconstructing its tissue domains and boundary sets."
        ),
        epilog="See work/README_uniform_remesh.md for requirements, assumptions, and validation guidance.",
    )
    parser.add_argument("input", type=Path, help="source FEBio .feb model")
    parser.add_argument("output", type=Path, help="destination .feb model; use a new filename")
    parser.add_argument("--target", type=float, default=2.0, help="target edge length in mm")
    parser.add_argument(
        "--vendor",
        type=Path,
        default=Path(__file__).resolve().parent / ".gmsh-venv" / "Lib" / "site-packages",
        help="directory containing gmsh.py (defaults to work/.gmsh-venv/Lib/site-packages)",
    )
    return parser.parse_args()


def load_feb(path: Path):
    root = ET.parse(path).getroot()
    mesh = root.find("Mesh")
    if mesh is None:
        raise ValueError("Input has no Mesh section")
    nodes_block = mesh.find("Nodes")
    if nodes_block is None:
        raise ValueError("Input has no Nodes block")
    node_ids = np.array([int(n.attrib["id"]) for n in nodes_block], dtype=np.int64)
    points = np.array([np.fromstring(n.text or "", sep=",") for n in nodes_block], dtype=float)
    id_to_row = {int(tag): i for i, tag in enumerate(node_ids)}

    tet_blocks = []
    tet_regions = []
    for region_index, name in enumerate(REGIONS):
        block = next((b for b in mesh.findall("Elements") if b.attrib.get("name") == name), None)
        if block is None:
            raise ValueError(f"Missing tet block {name!r}")
        conn_ids = np.array([[int(v) for v in (e.text or "").split(",")] for e in block], dtype=np.int64)
        conn = np.array([[id_to_row[int(v)] for v in row] for row in conn_ids], dtype=np.int64)
        tet_blocks.append(conn)
        tet_regions.append(np.full(len(conn), region_index, dtype=np.int8))
    tets = np.vstack(tet_blocks)
    regions = np.concatenate(tet_regions)

    node_sets = {}
    for node_set in mesh.findall("NodeSet"):
        raw = (node_set.text or "").replace("\n", "")
        node_sets[node_set.attrib["name"]] = [int(v) for v in raw.split(",") if v.strip()]
    return points, node_ids, id_to_row, tets, regions, node_sets


def outward_face(points: np.ndarray, face: tuple[int, int, int], opposite: int) -> tuple[int, int, int]:
    a, b, c = (points[i] for i in face)
    if np.dot(np.cross(b - a, c - a), points[opposite] - a) > 0:
        return face[0], face[2], face[1]
    return face


def boundary_faces(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    faces: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    local = ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3))
    for tet in tets:
        for i, j, k, opp in local:
            raw = (int(tet[i]), int(tet[j]), int(tet[k]))
            key = tuple(sorted(raw))
            if key in faces:
                faces[key] = None
            else:
                faces[key] = outward_face(points, raw, int(tet[opp]))
    return np.array([face for face in faces.values() if face is not None], dtype=np.int64)


def write_binary_stl(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(b"FEBio isotropic remesh surface".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(faces)))
        for face in faces:
            a, b, c = points[face]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            if length:
                normal /= length
            values = np.concatenate((normal, a, b, c)).astype(np.float32)
            stream.write(struct.pack("<12fH", *values, 0))


def generate_gmsh(stl_path: Path, target: float, vendor: Path):
    sys.path.insert(0, str(vendor.resolve()))
    import gmsh  # type: ignore

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("uniform_breast_remesh")
        gmsh.merge(str(stl_path.resolve()))
        # Treat the segmented exterior as one smooth anatomical envelope.  A
        # feature-angle split would preserve thousands of voxel stair-step edges
        # and can create non-parametrizable micro-patches on this source mesh.
        gmsh.model.mesh.classifySurfaces(math.pi, True, True, math.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError("Gmsh did not create any closed surface patches")
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", target)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)
        tag_to_row = {int(tag): i for i, tag in enumerate(node_tags)}

        types, _element_tags, element_nodes = gmsh.model.mesh.getElements(3)
        tet_conn = []
        for element_type, flat in zip(types, element_nodes):
            if int(element_type) != 4:
                continue
            tags = np.asarray(flat, dtype=np.int64).reshape(-1, 4)
            tet_conn.append(np.array([[tag_to_row[int(v)] for v in row] for row in tags], dtype=np.int64))
        if not tet_conn:
            raise RuntimeError("Gmsh produced no linear tetrahedra")
        return coords, np.vstack(tet_conn)
    finally:
        gmsh.finalize()


def triangle_quality(points: np.ndarray, faces: np.ndarray):
    tri = points[faces]
    lengths = np.stack(
        (
            np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
        ),
        axis=1,
    )
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    quality = 4 * math.sqrt(3) * area / np.sum(lengths * lengths, axis=1)
    return quality, lengths.ravel()


def tet_quality(points: np.ndarray, tets: np.ndarray):
    xyz = points[tets]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]),
        xyz[:, 3] - xyz[:, 0],
    )
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_sq_sum = sum(np.sum((xyz[:, i] - xyz[:, j]) ** 2, axis=1) for i, j in edges)
    volume = np.abs(signed_six_volume) / 6
    quality = 12 * np.power(3 * volume, 2 / 3) / edge_sq_sum
    return quality, signed_six_volume


def wrap_ids(values, indent="      ", per_line=18):
    values = [str(int(v)) for v in values]
    return ",\n".join(indent + ",".join(values[i : i + per_line]) for i in range(0, len(values), per_line))


def build_mesh_xml(
    points: np.ndarray,
    tets: np.ndarray,
    region_ids: np.ndarray,
    skin_faces: np.ndarray,
    chest_nodes: list[int],
    anchors: tuple[int, int],
) -> str:
    lines = ["  <Mesh>", '    <Nodes name="breast_nodes">']
    for i, xyz in enumerate(points, 1):
        lines.append(f'      <node id="{i}">{xyz[0]:.10g},{xyz[1]:.10g},{xyz[2]:.10g}</node>')
    lines.append("    </Nodes>")
    element_id = 1
    for region_index, name in enumerate(REGIONS):
        lines.append(f'    <Elements type="tet4" name="{name}">')
        for tet in tets[region_ids == region_index]:
            conn = ",".join(str(int(v) + 1) for v in tet)
            lines.append(f'      <elem id="{element_id}">{conn}</elem>')
            element_id += 1
        lines.append("    </Elements>")
    lines.extend(
        (
            "    <!-- Uniform isotropic exterior shell; ChestWall faces are omitted. -->",
            '    <Elements type="tri3" name="skin">',
        )
    )
    for face in skin_faces:
        conn = ",".join(str(int(v) + 1) for v in face)
        lines.append(f'      <elem id="{element_id}">{conn}</elem>')
        element_id += 1
    lines.extend(
        (
            "    </Elements>",
            '    <NodeSet name="ChestWall">',
            wrap_ids(chest_nodes),
            "    </NodeSet>",
            f'    <NodeSet name="PosteriorAnchorA">{anchors[0]}</NodeSet>',
            f'    <NodeSet name="PosteriorAnchorB">{anchors[1]}</NodeSet>',
            "  </Mesh>",
        )
    )
    return "\n".join(lines)


def build_mesh_data_xml(skin_count: int) -> str:
    lines = [
        "  <MeshData>",
        "    <!-- Constant membrane-equivalent numerical skin thickness. -->",
        '    <ElementData type="shell thickness" elem_set="skin">',
    ]
    lines.extend(f'      <elem lid="{i}">0.001,0.001,0.001</elem>' for i in range(1, skin_count + 1))
    lines.extend(("    </ElementData>", "  </MeshData>"))
    return "\n".join(lines)


def replace_section(text: str, tag: str, replacement: str) -> str:
    start = text.index(f"  <{tag}>")
    end = text.index(f"  </{tag}>", start) + len(f"  </{tag}>")
    return text[:start] + replacement + text[end:]


def main() -> None:
    args = parse_args()
    points, node_ids, id_to_row, old_tets, old_regions, node_sets = load_feb(args.input)
    old_boundary = boundary_faces(points, old_tets)
    chest_old_ids = set(node_sets["ChestWall"])
    chest_old_rows = {id_to_row[v] for v in chest_old_ids}
    old_chest_faces = np.array(
        [face for face in old_boundary if all(int(v) in chest_old_rows for v in face)], dtype=np.int64
    )
    old_skin_faces = np.array(
        [face for face in old_boundary if not all(int(v) in chest_old_rows for v in face)], dtype=np.int64
    )
    if not len(old_chest_faces):
        raise RuntimeError("No exterior ChestWall faces were found")

    cache_dir = args.output.parent / ".remesh-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stl_path = cache_dir / f"{args.output.stem}-outer-surface.stl"
    write_binary_stl(stl_path, points, old_boundary)
    print(f"Old mesh: {len(points):,} nodes, {len(old_tets):,} tets, {len(old_boundary):,} exterior faces")
    print(f"Gmsh isotropic target: {args.target:g} mm")
    new_points, new_tets = generate_gmsh(stl_path, args.target, args.vendor)
    new_boundary = boundary_faces(new_points, new_tets)

    new_centroids = new_points[new_tets].mean(axis=1)
    # Use an exact containing-cell lookup for material transfer.  Only points
    # that fall just outside because of floating-point projection use the
    # nearest source centroid as a fallback.
    import pyvista as pv

    old_grid = pv.UnstructuredGrid({pv.CellType.TETRA: old_tets}, points)
    containing_cell = old_grid.find_containing_cell(new_centroids)
    new_regions = np.empty(len(new_tets), dtype=np.int8)
    inside = containing_cell >= 0
    new_regions[inside] = old_regions[containing_cell[inside]]
    if np.any(~inside):
        old_centroids = points[old_tets].mean(axis=1)
        _, nearest_old_tet = cKDTree(old_centroids).query(new_centroids[~inside], workers=-1)
        new_regions[~inside] = old_regions[nearest_old_tet]

    old_chest_centroids = points[old_chest_faces].mean(axis=1)
    old_skin_centroids = points[old_skin_faces].mean(axis=1)
    new_face_centroids = new_points[new_boundary].mean(axis=1)
    chest_distance, _ = cKDTree(old_chest_centroids).query(new_face_centroids, workers=-1)
    skin_distance, _ = cKDTree(old_skin_centroids).query(new_face_centroids, workers=-1)
    is_chest_face = (chest_distance < skin_distance) & (chest_distance <= 2.0 * args.target)
    chest_faces = new_boundary[is_chest_face]
    skin_faces = new_boundary[~is_chest_face]
    chest_rows = sorted(set(int(v) for v in chest_faces.ravel()))
    chest_node_ids = [v + 1 for v in chest_rows]

    anchor_old_coords = []
    for name in ("PosteriorAnchorA", "PosteriorAnchorB"):
        old_id = node_sets[name][0]
        anchor_old_coords.append(points[id_to_row[old_id]])
    chest_tree = cKDTree(new_points[chest_rows])
    _, anchor_local = chest_tree.query(np.array(anchor_old_coords))
    anchor_rows = [chest_rows[int(i)] for i in anchor_local]
    if anchor_rows[0] == anchor_rows[1]:
        distances = np.linalg.norm(new_points[chest_rows] - anchor_old_coords[1], axis=1)
        for candidate in np.argsort(distances):
            if chest_rows[int(candidate)] != anchor_rows[0]:
                anchor_rows[1] = chest_rows[int(candidate)]
                break
    anchors = (anchor_rows[0] + 1, anchor_rows[1] + 1)

    tri_q, tri_edges = triangle_quality(new_points, skin_faces)
    tet_q, signed_six_volume = tet_quality(new_points, new_tets)
    if np.any(np.abs(signed_six_volume) < 1e-12):
        raise RuntimeError("Remesh contains zero-volume tetrahedra")
    # FEBio accepts consistently oriented tet4 elements. Reorient negative entries.
    negative = signed_six_volume < 0
    if np.any(negative):
        new_tets[negative, 0], new_tets[negative, 1] = (
            new_tets[negative, 1].copy(),
            new_tets[negative, 0].copy(),
        )

    mesh_xml = build_mesh_xml(new_points, new_tets, new_regions, skin_faces, chest_node_ids, anchors)
    mesh_data_xml = build_mesh_data_xml(len(skin_faces))
    original_text = args.input.read_text(encoding="utf-8")
    output_text = replace_section(original_text, "Mesh", mesh_xml)
    output_text = replace_section(output_text, "MeshData", mesh_data_xml)
    output_text = output_text.replace(args.input.stem, args.output.stem)
    marker = "  Uniform isotropic remesh generated with Gmsh:\n"
    marker += f"  - target surface/volume edge length: {args.target:g} mm;\n"
    marker += "  - material labels transferred by source-cell containment with centroid fallback;\n"
    marker += "  - ChestWall and posterior anchors reconstructed geometrically.\n"
    output_text = output_text.replace("<!--\n", "<!--\n" + marker, 1)
    args.output.write_text(output_text, encoding="utf-8", newline="\n")

    quality_path = args.output.with_suffix(".quality.txt")
    counts = {REGIONS[i]: int(np.sum(new_regions == i)) for i in range(len(REGIONS))}
    report = "\n".join(
        (
            f"input={args.input}",
            f"output={args.output}",
            f"target_edge_mm={args.target:g}",
            f"nodes={len(new_points)}",
            f"tets={len(new_tets)}",
            *(f"{name}_tets={counts[name]}" for name in REGIONS),
            f"skin_triangles={len(skin_faces)}",
            f"chestwall_faces={len(chest_faces)}",
            f"chestwall_nodes={len(chest_node_ids)}",
            f"anchors={anchors[0]},{anchors[1]}",
            "surface_edge_mm_percentiles=" + ",".join(f"{v:.6g}" for v in np.percentile(tri_edges, [0, 1, 5, 50, 95, 99, 100])),
            "surface_quality_percentiles=" + ",".join(f"{v:.6g}" for v in np.percentile(tri_q, [0, 1, 5, 50, 95, 99, 100])),
            f"surface_triangles_quality_lt_0.7={int(np.sum(tri_q < 0.7))}",
            "tet_quality_percentiles=" + ",".join(f"{v:.6g}" for v in np.percentile(tet_q, [0, 1, 5, 50, 95, 99, 100])),
        )
    ) + "\n"
    quality_path.write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
