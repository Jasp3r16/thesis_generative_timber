"""
GHPython component script: bake top-10 design elements to Rhino layers.

Add these inputs in the GH component editor:
  geometry_ns : list  — curves/lines for new-stock members
  geometry_rs : list  — curves/lines for reclaimed-stock members
  color_ns    : Colour — object + layer colour for NS (default: steel blue)
  color_rs    : Colour — object + layer colour for RS (default: ochre orange)
  rank        : int   — slider 1–10, which design rank to bake
  bake        : bool  — button, triggers bake on True
  clear       : bool  — if True, deletes the existing rank layer first

Outputs:
  status      : string — result message
  debug       : string — type info for the first received item (disconnect once working)
"""

import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System.Drawing


# =============================================================================
# Helpers
# =============================================================================

_DEFAULT_COLOR_NS = System.Drawing.Color.FromArgb(70, 130, 180)   # steel blue
_DEFAULT_COLOR_RS = System.Drawing.Color.FromArgb(210, 105, 30)   # ochre orange


def _as_list(data):
    """Flatten GH DataTree or plain list to a Python list."""
    if data is None:
        return []
    if hasattr(data, "BranchCount") and hasattr(data, "Branch"):
        out = []
        for i in range(data.BranchCount):
            out.extend(list(data.Branch(i)))
        return out
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def _resolve_color(gh_color, default):
    """Convert a GH Colour input to System.Drawing.Color, falling back to default."""
    if gh_color is None:
        return default
    # GH passes colours as System.Drawing.Color already
    if isinstance(gh_color, System.Drawing.Color):
        return gh_color
    # Some GH versions pass a list; take first element
    if hasattr(gh_color, "__iter__"):
        for item in gh_color:
            if isinstance(item, System.Drawing.Color):
                return item
    return default


def _find_layer(full_path):
    """Return layer index or -1 in sc.doc."""
    return sc.doc.Layers.FindByFullPath(full_path, True)


def _ensure_layer(name, parent_index=-1, color=None):
    """Return layer index in sc.doc, creating the layer if necessary."""
    if parent_index >= 0:
        full_path = "{}::{}".format(sc.doc.Layers[parent_index].FullPath, name)
    else:
        full_path = name

    idx = _find_layer(full_path)
    if idx >= 0:
        if color is not None:
            sc.doc.Layers[idx].Color = color
        return idx

    layer = Rhino.DocObjects.Layer()
    layer.Name = name
    if color is not None:
        layer.Color = color
    if parent_index >= 0:
        layer.ParentLayerId = sc.doc.Layers[parent_index].Id
    return sc.doc.Layers.Add(layer)


def _clear_layer_tree(layer_full_path):
    """Delete all objects and sublayers under layer_full_path using rhinoscriptsyntax."""
    if not rs.IsLayer(layer_full_path):
        return

    all_names = []

    def _collect(name):
        all_names.append(name)
        for child in rs.LayerChildren(name) or []:
            _collect(child)

    _collect(layer_full_path)

    for name in reversed(all_names):
        objs = rs.ObjectsByLayer(name)
        if objs:
            rs.DeleteObjects(objs)

    for name in reversed(all_names):
        if rs.IsLayer(name):
            rs.DeleteLayer(name)


def _to_geom_base(item, ghdoc):
    """Extract GeometryBase from a GH input item.

    GUIDs must be resolved from ghdoc (the GH volatile doc) before sc.doc is
    switched to the Rhino doc for baking.
    """
    if item is None:
        return None
    if isinstance(item, Rhino.Geometry.GeometryBase):
        return item
    if type(item).__name__ == "Guid":
        obj = ghdoc.Objects.Find(item)
        return obj.Geometry.Duplicate() if obj else None
    if isinstance(item, Rhino.DocObjects.ObjRef):
        g = item.Geometry()
        return g.Duplicate() if g else None
    if hasattr(item, "Value"):
        val = item.Value
        if isinstance(val, Rhino.Geometry.GeometryBase):
            return val
    return None


def _bake_list(geom_list, layer_idx, obj_color):
    """Add pre-resolved GeometryBase items to sc.doc (must be set to Rhino doc)."""
    count = 0
    for geom in geom_list:
        if geom is None:
            continue
        attr = Rhino.DocObjects.ObjectAttributes()
        attr.LayerIndex = layer_idx
        attr.ObjectColor = obj_color
        attr.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
        sc.doc.Objects.Add(geom, attr)
        count += 1
    return count


# =============================================================================
# Main
# =============================================================================

status = "Idle"
debug  = ""

ns_geom     = _as_list(geometry_ns)                            # noqa: F821
rs_geom     = _as_list(geometry_rs)                            # noqa: F821
run_flag    = bool(bake)                                        # noqa: F821
clear_flag  = bool(clear) if clear is not None else False      # noqa: F821
design_rank = int(rank) if rank is not None else 1             # noqa: F821
col_ns      = _resolve_color(color_ns, _DEFAULT_COLOR_NS)      # noqa: F821
col_rs      = _resolve_color(color_rs, _DEFAULT_COLOR_RS)      # noqa: F821

# Always emit type info for the first item regardless of bake flag
_sample = (ns_geom or rs_geom or [None])[0]
if _sample is not None:
    debug = "type={} | dir={}".format(
        type(_sample).__name__,
        [a for a in dir(_sample) if not a.startswith("_")][:12],
    )
else:
    debug = "ns={} rs={} — no items received".format(len(ns_geom), len(rs_geom))

if not run_flag:
    status = "bake=False — nothing baked"

elif not ns_geom and not rs_geom:
    status = "No geometry received on geometry_ns or geometry_rs"

else:
    root_name   = "top10_designs"
    design_name = "design_{}".format(design_rank)
    design_path = "{}::{}".format(root_name, design_name)  # full path for clear/status

    # Resolve all GH geometry BEFORE switching sc.doc away from ghdoc
    ghdoc    = sc.doc
    ns_geoms = [_to_geom_base(item, ghdoc) for item in ns_geom]
    rs_geoms = [_to_geom_base(item, ghdoc) for item in rs_geom]

    # Switch sc.doc to Rhino doc for all layer + object operations
    sc.doc = Rhino.RhinoDoc.ActiveDoc
    try:
        if clear_flag:
            _clear_layer_tree(design_path)

        root_idx   = _ensure_layer(root_name)
        design_idx = _ensure_layer(design_name, parent_index=root_idx)
        ns_idx     = _ensure_layer("NS", parent_index=design_idx, color=col_ns)
        rs_idx     = _ensure_layer("RS", parent_index=design_idx, color=col_rs)

        n_ns = _bake_list(ns_geoms, ns_idx, col_ns)
        n_rs = _bake_list(rs_geoms, rs_idx, col_rs)

        sc.doc.Views.Redraw()
        status = "Design #{}: baked {} NS + {} RS → '{}'".format(
            design_rank, n_ns, n_rs, design_path
        )
    finally:
        sc.doc = ghdoc  # always restore GH doc